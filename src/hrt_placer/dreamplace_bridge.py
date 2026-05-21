"""Bridge between the challenge ``Benchmark`` and DREAMPlace.

The challenge benchmark is a parsed protobuf netlist; DREAMPlace consumes the
Bookshelf format via a compiled C++ reader. This module:

1. ``write_bookshelf`` -- writes a Benchmark as a Bookshelf design
   (.aux/.nodes/.nets/.pl/.scl/.wts) and returns the node-name -> macro-index
   map needed to read the result back.
2. ``parse_pl`` -- parses a DREAMPlace ``.pl`` solution back into a
   ``[num_macros, 2]`` array of macro *centres* in micron coordinates.

Design choice: we use DREAMPlace for **global placement only** -- the
electrostatic spreading that our own engine approximates poorly. DREAMPlace's
row legalisation / detailed placement is skipped; the caller re-legalises the
global solution with the project's own push-apart + shelf-pack legaliser,
which already guarantees zero hard-macro overlap and the >=12 um clearance.
That keeps the Bookshelf ``.scl`` minimal (DREAMPlace only needs the region
bounding box for global placement) and avoids a large class of
row-legalisation bridge bugs.

Coordinates: the Benchmark is in microns (floats); Bookshelf is integer-grid.
A fixed integer ``SCALE`` maps microns -> Bookshelf units; the inverse is
applied when parsing the solution. Node names are ``m{i}`` for macros and
``p{j}`` for ports, so the solution maps straight back to macro indices.
"""

import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import torch

# Microns -> Bookshelf integer units. Large enough that sub-micron geometry
# survives rounding; small enough to stay well within int32.
SCALE = 100


def _ll(centre, size):
    """Centre coords + size -> lower-left corner (Bookshelf node origin)."""
    return centre - size / 2.0


def write_bookshelf(benchmark, outdir, design="design"):
    """Write ``benchmark`` as a Bookshelf design under ``outdir``.

    Returns ``(aux_path, name_to_index)`` where ``name_to_index`` maps every
    Bookshelf node name back to its challenge macro index (ports excluded --
    ports are fixed terminals and are not part of the returned placement).
    """
    os.makedirs(outdir, exist_ok=True)
    nm = benchmark.num_macros
    nh = benchmark.num_hard_macros
    pos = benchmark.macro_positions.detach().cpu().numpy().astype(np.float64)
    size = benchmark.macro_sizes.detach().cpu().numpy().astype(np.float64)
    fixed = benchmark.macro_fixed.detach().cpu().numpy().astype(bool)
    ports = benchmark.port_positions.detach().cpu().numpy().astype(np.float64)
    n_ports = ports.shape[0]
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)

    macro_names = [f"m{i}" for i in range(nm)]
    port_names = [f"p{j}" for j in range(n_ports)]
    name_to_index = {macro_names[i]: i for i in range(nm)}

    # ---- .nodes -----------------------------------------------------------
    # Bookshelf node line: "<name> <w> <h>" (+ " terminal" if fixed).
    n_terminals = int(fixed.sum()) + n_ports
    lines = ["UCLA nodes 1.0", "",
             f"NumNodes : {nm + n_ports}",
             f"NumTerminals : {n_terminals}", ""]
    for i in range(nm):
        w = max(1, round(size[i, 0] * SCALE))
        h = max(1, round(size[i, 1] * SCALE))
        suffix = " terminal" if fixed[i] else ""
        lines.append(f"\t{macro_names[i]}\t{w}\t{h}{suffix}")
    for j in range(n_ports):
        # Ports are dimensionless pins; give them a 1x1 terminal footprint.
        lines.append(f"\t{port_names[j]}\t1\t1 terminal")
    with open(os.path.join(outdir, f"{design}.nodes"), "w") as f:
        f.write("\n".join(lines) + "\n")

    # ---- .pl --------------------------------------------------------------
    # Bookshelf placement line: "<name> <x> <y> : <orient> [/FIXED]".
    # x, y are the lower-left corner.
    lines = ["UCLA pl 1.0", ""]
    for i in range(nm):
        llx = round(_ll(pos[i, 0], size[i, 0]) * SCALE)
        lly = round(_ll(pos[i, 1], size[i, 1]) * SCALE)
        suffix = " /FIXED" if fixed[i] else ""
        lines.append(f"{macro_names[i]}\t{llx}\t{lly}\t: N{suffix}")
    for j in range(n_ports):
        px = round(ports[j, 0] * SCALE)
        py = round(ports[j, 1] * SCALE)
        lines.append(f"{port_names[j]}\t{px}\t{py}\t: N /FIXED")
    with open(os.path.join(outdir, f"{design}.pl"), "w") as f:
        f.write("\n".join(lines) + "\n")

    # ---- .nets ------------------------------------------------------------
    # Prefer pin-level connectivity; fall back to per-macro net_nodes.
    net_lines = []
    total_pins = 0
    use_pin = (len(benchmark.net_pin_nodes) == benchmark.num_nets
               and benchmark.num_nets > 0)
    pin_offsets = [o.detach().cpu().numpy().astype(np.float64)
                   for o in benchmark.macro_pin_offsets]
    for net_id in range(benchmark.num_nets):
        if use_pin:
            pins = benchmark.net_pin_nodes[net_id].detach().cpu().numpy()
        else:
            nodes = benchmark.net_nodes[net_id].detach().cpu().numpy()
            pins = np.stack([nodes, np.zeros_like(nodes)], axis=1)
        if pins.shape[0] < 2:
            continue
        body = []
        for owner, pidx in pins:
            owner = int(owner)
            pidx = int(pidx)
            if owner < nm:
                name = macro_names[owner]
                ox = oy = 0.0
                if owner < nh and owner < len(pin_offsets):
                    off = pin_offsets[owner]
                    if 0 <= pidx < off.shape[0]:
                        ox, oy = float(off[pidx, 0]), float(off[pidx, 1])
                body.append(f"\t{name} B : "
                            f"{round(ox * SCALE)} {round(oy * SCALE)}")
            else:
                pj = owner - nm
                if 0 <= pj < n_ports:
                    body.append(f"\t{port_names[pj]} B : 0 0")
        if len(body) < 2:
            continue
        total_pins += len(body)
        net_lines.append(f"NetDegree : {len(body)} n{net_id}")
        net_lines.extend(body)
    header = ["UCLA nets 1.0", "",
              f"NumNets : {sum(1 for L in net_lines if L.startswith('NetDegree'))}",
              f"NumPins : {total_pins}", ""]
    with open(os.path.join(outdir, f"{design}.nets"), "w") as f:
        f.write("\n".join(header + net_lines) + "\n")

    # ---- .scl -------------------------------------------------------------
    # Minimal row structure: tile the canvas with unit-height rows. Global
    # placement only needs the region bounding box; the caller re-legalises.
    cw_u = round(cw * SCALE)
    ch_u = round(ch * SCALE)
    row_h = max(1, ch_u // 256)  # ~256 rows; fine for global placement
    n_rows = max(1, ch_u // row_h)
    scl = ["UCLA scl 1.0", "", f"NumRows : {n_rows}", ""]
    for r in range(n_rows):
        scl += [
            "CoreRow Horizontal",
            f"  Coordinate    :   {r * row_h}",
            f"  Height        :   {row_h}",
            "  Sitewidth     :   1",
            "  Sitespacing   :   1",
            "  Siteorient    :   N",
            "  Sitesymmetry  :   Y",
            f"  SubrowOrigin  :   0  NumSites  :  {cw_u}",
            "End",
        ]
    with open(os.path.join(outdir, f"{design}.scl"), "w") as f:
        f.write("\n".join(scl) + "\n")

    # ---- .wts (net weights) ----------------------------------------------
    with open(os.path.join(outdir, f"{design}.wts"), "w") as f:
        f.write("UCLA wts 1.0\n")

    # ---- .aux -------------------------------------------------------------
    aux_path = os.path.join(outdir, f"{design}.aux")
    with open(aux_path, "w") as f:
        f.write(f"RowBasedPlacement : {design}.nodes {design}.nets "
                f"{design}.wts {design}.pl {design}.scl\n")

    return aux_path, name_to_index


def parse_pl(pl_file, name_to_index, num_macros, benchmark):
    """Parse a DREAMPlace ``.pl`` solution into ``[num_macros, 2]`` centres.

    DREAMPlace writes lower-left corners in Bookshelf units; this converts
    back to micron centres. Macros missing from the solution (should not
    happen) keep their original benchmark position.
    """
    size = benchmark.macro_sizes.detach().cpu().numpy().astype(np.float64)
    out = benchmark.macro_positions.detach().cpu().numpy().astype(np.float64).copy()
    seen = set()
    with open(pl_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("UCLA") or line.startswith("#"):
                continue
            tok = line.split()
            if len(tok) < 3:
                continue
            name = tok[0]
            idx = name_to_index.get(name)
            if idx is None:
                continue
            try:
                llx = float(tok[1]) / SCALE
                lly = float(tok[2]) / SCALE
            except ValueError:
                continue
            # lower-left -> centre
            out[idx, 0] = llx + size[idx, 0] / 2.0
            out[idx, 1] = lly + size[idx, 1] / 2.0
            seen.add(idx)
    return torch.from_numpy(out).float(), len(seen)


def write_config(aux_path, result_dir, design="design", gpu=True,
                 target_density=0.5):
    """Write a DREAMPlace JSON config for the full place flow.

    The TILOS proxy punishes bin density, so ``target_density`` is set low
    (0.5) to make DREAMPlace spread macros rather than pack them. legalize +
    detailed placement are ON -- a global-placement-only run leaves macros
    overlapping at high density (global placement allows overlap by design;
    DREAMPlace's legalize/detailed steps are what spread it). macro_place_flag
    enables the 2-stage macro flow. Keys not set here fall back to
    params.json defaults.
    """
    cfg = {
        "aux_input": aux_path,
        "result_dir": result_dir,
        "gpu": 1 if gpu else 0,
        "global_place_flag": 1,
        "legalize_flag": 1,
        "detailed_place_flag": 1,
        "macro_place_flag": 1,
        "target_density": target_density,
        "stop_overflow": 0.07,
        "enable_fillers": 1,
        "dtype": "float32",
        "random_seed": 1000,
    }
    cfg_path = os.path.join(result_dir, f"{design}.json")
    os.makedirs(result_dir, exist_ok=True)
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)
    return cfg_path


def run_dreamplace(benchmark, dreamplace_root, work_dir=None, gpu=True,
                   timeout=1800.0):
    """Run DREAMPlace global placement on ``benchmark``.

    Writes the Bookshelf design + config under ``work_dir`` (a fresh temp dir
    if None), invokes ``dreamplace/Placer.py`` as a subprocess, parses the
    ``.gp.pl`` solution. Returns ``[num_macros, 2]`` micron centres, or raises
    on any failure (the caller falls back to the analytical engine).

    ``dreamplace_root`` is the DREAMPlace repo root (contains ``dreamplace/``).
    """
    if work_dir is None:
        work_dir = tempfile.mkdtemp(prefix="hrt_dp_")
    design = "design"
    aux_path, name_to_index = write_bookshelf(benchmark, work_dir, design)
    result_dir = os.path.join(work_dir, "results")
    cfg_path = write_config(aux_path, result_dir, design, gpu=gpu)

    placer_py = os.path.join(dreamplace_root, "dreamplace", "Placer.py")
    if not os.path.exists(placer_py):
        raise FileNotFoundError(f"DREAMPlace not found at {placer_py}")

    env = dict(os.environ)
    env["PYTHONPATH"] = (dreamplace_root + os.pathsep
                         + env.get("PYTHONPATH", ""))
    proc = subprocess.run(
        [sys.executable, placer_py, cfg_path],
        cwd=dreamplace_root, env=env, timeout=timeout,
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "")[-2000:]
        raise RuntimeError(f"DREAMPlace exited {proc.returncode}: {tail}")

    gp_pl = os.path.join(result_dir, design, f"{design}.gp.pl")
    if not os.path.exists(gp_pl):
        raise FileNotFoundError(f"DREAMPlace produced no solution at {gp_pl}")

    centres, n_seen = parse_pl(gp_pl, name_to_index, benchmark.num_macros,
                               benchmark)
    if n_seen < benchmark.num_macros * 0.5:
        raise RuntimeError(f"DREAMPlace solution covered only {n_seen}/"
                           f"{benchmark.num_macros} macros")
    return centres
