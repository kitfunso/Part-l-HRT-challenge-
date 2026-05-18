"""Calibrate our ProxyCost reimplementation against the real TILOS evaluator.

For each benchmark, scores several placements (reference, greedy, analytical)
with both the challenge evaluator and our from-Benchmark ProxyCost, then reports
per-term error and rank correlation. Run inside the challenge venv:

    cd external/macro-place-challenge-2026
    uv run python ../../scripts/calibrate.py ibm01 ibm03 ibm07 ibm09 ibm12
"""

import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHALLENGE = os.path.join(REPO, "external", "macro-place-challenge-2026")
sys.path.insert(0, os.path.join(REPO, "src"))
os.chdir(CHALLENGE)

import torch  # noqa: E402

from macro_place.loader import load_benchmark_from_dir  # noqa: E402
from macro_place.objective import compute_proxy_cost  # noqa: E402

from hrt_placer.engine import AnalyticalPlacer  # noqa: E402
from hrt_placer.proxy_cost import ProxyCost  # noqa: E402

BASE = "external/MacroPlacement/Testcases/ICCAD04"


def greedy(benchmark):
    placement = benchmark.macro_positions.clone()
    movable = benchmark.get_movable_mask() & benchmark.get_hard_macro_mask()
    idx = torch.where(movable)[0].tolist()
    sizes = benchmark.macro_sizes
    idx.sort(key=lambda i: -sizes[i, 1].item())
    cx = cy = rh = 0.0
    for i in idx:
        w, h = sizes[i, 0].item(), sizes[i, 1].item()
        if cx + w > benchmark.canvas_width:
            cx = 0.0
            cy += rh + 0.001
            rh = 0.0
        if cy + h <= benchmark.canvas_height:
            placement[i, 0] = cx + w / 2
            placement[i, 1] = cy + h / 2
            cx += w + 0.001
            rh = max(rh, h)
    return placement


def main():
    names = sys.argv[1:] or ["ibm01", "ibm03", "ibm07", "ibm09", "ibm12"]
    pairs = []  # (real_proxy, our_proxy)
    print(f"{'bench':>7} {'variant':>10} | "
          f"{'real':>8} {'wl':>7} {'den':>7} {'cong':>7} | "
          f"{'ours':>8} {'wl':>7} {'den':>7} {'cong':>7} | {'dProxy':>8}")
    for name in names:
        bm, plc = load_benchmark_from_dir(f"{BASE}/{name}")
        pc = ProxyCost(bm)
        variants = {
            "reference": bm.macro_positions.clone(),
            "greedy": greedy(bm),
            "analytical": AnalyticalPlacer().place(bm),
        }
        for vname, placement in variants.items():
            real = compute_proxy_cost(placement, bm, plc)
            ours = pc(placement)
            pairs.append((real["proxy_cost"], ours["proxy_cost"]))
            d = ours["proxy_cost"] - real["proxy_cost"]
            print(f"{name:>7} {vname:>10} | "
                  f"{real['proxy_cost']:8.4f} {real['wirelength_cost']:7.3f} "
                  f"{real['density_cost']:7.3f} {real['congestion_cost']:7.3f} | "
                  f"{ours['proxy_cost']:8.4f} {ours['wirelength_cost']:7.3f} "
                  f"{ours['density_cost']:7.3f} {ours['congestion_cost']:7.3f} | "
                  f"{d:+8.4f}")

    if len(pairs) >= 2:
        r = torch.tensor([p[0] for p in pairs])
        o = torch.tensor([p[1] for p in pairs])
        rc = float(((r - r.mean()) * (o - o.mean())).sum()
                   / (r.std(unbiased=False) * o.std(unbiased=False)
                      * len(r) + 1e-12))
        # Spearman rank correlation
        rr = r.argsort().argsort().float()
        oo = o.argsort().argsort().float()
        sp = float(((rr - rr.mean()) * (oo - oo.mean())).sum()
                   / (rr.std(unbiased=False) * oo.std(unbiased=False)
                      * len(rr) + 1e-12))
        mae = float((r - o).abs().mean())
        print(f"\n  n={len(pairs)}  Pearson={rc:.4f}  Spearman={sp:.4f}  "
              f"MAE={mae:.4f}")


if __name__ == "__main__":
    main()
