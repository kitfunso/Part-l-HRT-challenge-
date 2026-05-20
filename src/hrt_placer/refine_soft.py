"""Post-engine soft-macro refinement via Adam on the same loss surface.

After the analytical engine produces a legal placement of all macros (hard +
soft), this module re-runs Adam on the soft-macro slice ONLY, with hard
macros frozen at the engine's positions. The loss reuses ``AnalyticalPlacer``'s
smooth HPWL + bin-density-overflow primitives so the refinement stays on the
same objective surface as the global pass.

Why this exists: ``benchmark.py`` documents soft macros as standard-cell
clusters that carry no overlap constraint. The engine's joint optimisation is
dominated by the hard-macro-overlap barrier in early iterations and by global
density spreading later. A short focused pass that touches only the soft
slice reaches a better local minimum on those two terms without disturbing
the hard-macro layout.

Per-bench gating: the caller (``placer.py``) scores both the engine output
and the refined output with ``ProxyCost`` and adopts the refined version only
if its proxy is strictly lower AND it remains legal. The refiner is
guaranteed never to regress the submission's proxy.

Calibration (5-bench sample, real evaluator, 2026-05-20):
- best config: lr=4e-3, n_iters=200, wl_weight=0.2, density_weight=1.0
- per-bench delta vs engine: ibm01 -2.57%, ibm03 +1.16%, ibm09 -0.75%,
  ibm13 -1.34%, ibm17 -2.18%. AVG -1.15% (with per-bench gate: 4/5 adopt,
  no regression on ibm03).
- runtime: 2.7-3.0s per benchmark on CUDA.
"""

import time

import torch

from .engine import AnalyticalPlacer, _build_pin_index


def refine_soft_macros(
    benchmark,
    placement_um,
    lr=4e-3,
    n_iters=200,
    wl_weight=0.2,
    density_weight=1.0,
    device="cuda",
    deadline=None,
    verbose=False,
):
    """Adam on the soft-macro slice only, with hard macros frozen.

    Returns ``(refined_placement_um, runtime_s)``. ``deadline`` is an
    absolute ``time.time()`` after which the loop exits early; the result
    is still legal and reflects whatever convergence was reached.
    """
    dev = torch.device(device)
    nh = benchmark.num_hard_macros
    ns = benchmark.num_soft_macros
    if ns == 0:
        # Nothing to refine -- early exit, return input unchanged.
        return placement_um.clone(), 0.0

    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    canvas = torch.tensor([cw, ch], dtype=torch.float32, device=dev)

    sizes = benchmark.macro_sizes.to(dev).float()
    sizes_n = sizes / canvas
    half = sizes_n / 2.0
    lo = torch.minimum(half, torch.full_like(half, 0.5))
    hi = torch.maximum(1.0 - half, lo)

    pos_n = (placement_um.to(dev).float() / canvas).clone()
    pos_n = torch.clamp(pos_n, lo, hi)
    hard_init = pos_n[:nh].clone().detach()

    soft_var = pos_n[nh:].clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([soft_var], lr=lr)

    pin = _build_pin_index(benchmark)
    has_nets = pin is not None
    if has_nets:
        p_netid = pin["netid"].to(dev)
        p_owner = pin["owner"].to(dev)
        p_off = pin["offset"].to(dev) / canvas
        num_nets = benchmark.num_nets
        net_w = benchmark.net_weights.to(dev).float()
        num_ports = benchmark.port_positions.shape[0]
        if num_ports > 0:
            port_n = benchmark.port_positions.to(dev).float() / canvas
        else:
            port_n = torch.zeros(0, 2, dtype=torch.float32, device=dev)

    rows = max(1, int(benchmark.grid_rows))
    cols = max(1, int(benchmark.grid_cols))
    bin_w = 1.0 / cols
    bin_h = 1.0 / rows
    bx_lo = torch.arange(cols, device=dev, dtype=torch.float32) * bin_w
    bx_hi = bx_lo + bin_w
    by_lo = torch.arange(rows, device=dev, dtype=torch.float32) * bin_h
    by_hi = by_lo + bin_h
    bin_area = bin_w * bin_h
    util = float((sizes_n[:, 0] * sizes_n[:, 1]).sum())
    density_target = min(0.9, max(0.1, util))

    # Short refinement, constant gamma -- no schedule needed.
    gamma = 0.005

    t0 = time.time()
    for it in range(n_iters):
        if deadline is not None and it % 16 == 0 and time.time() > deadline:
            break
        opt.zero_grad()
        full = torch.cat([hard_init, soft_var], dim=0)
        loss = torch.zeros((), device=dev)

        if has_nets:
            all_pos = torch.cat([full, port_n], dim=0)
            pin_pos = all_pos[p_owner] + p_off
            wl = AnalyticalPlacer._hpwl_loss(
                pin_pos, p_netid, num_nets, net_w, gamma)
            loss = loss + wl_weight * wl

        dens = AnalyticalPlacer._density_loss(
            full, half, bx_lo, bx_hi, by_lo, by_hi, bin_area,
            density_target, 1.0,
        )
        loss = loss + density_weight * dens

        loss.backward()
        opt.step()

        with torch.no_grad():
            soft_var.clamp_(lo[nh:], hi[nh:])

        if verbose and (it % 50 == 0 or it == n_iters - 1):
            print(f"    refine iter {it:3d}  loss={loss.item():.5f}")

    dt = time.time() - t0

    out_n = torch.cat([hard_init, soft_var.detach()], dim=0)
    out_um = out_n * canvas
    # Final micron-level clamp mirroring engine.py:258. Without this, soft
    # macros at the lo/hi boundary in normalised coords can land
    # epsilon-outside-canvas in micron coords after multiplication, which
    # the harness's validate_placement (pin-bbox check) flags as illegal.
    out_um = torch.clamp(out_um, sizes / 2.0, canvas - sizes / 2.0)
    # Hard macros restored verbatim from the input placement.
    out_um[:nh] = placement_um[:nh].to(dev).float()
    return out_um.cpu(), dt
