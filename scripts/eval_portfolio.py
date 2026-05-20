"""Quick A/B/C eval of the 3 engine portfolio configs.

Runs each ``_PORTFOLIO`` entry's ``AnalyticalPlacer`` on a benchmark sample
and prints proxy / wirelength / density / congestion for each. No SA. Lets
us see whether the new density_topk_frac and congestion_weight terms help
proxy before running the full MyPlacer pipeline on 17 benchmarks.

Usage (from inside ``external/macro-place-challenge-2026``):
    uv run python ../../scripts/eval_portfolio.py ibm01 ibm03 ibm09 ibm13 ibm17
"""

import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHALLENGE = os.path.join(REPO, "external", "macro-place-challenge-2026")
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, REPO)
sys.path.insert(0, CHALLENGE)
os.chdir(CHALLENGE)

from macro_place.loader import load_benchmark_from_dir  # noqa: E402
from macro_place.objective import compute_proxy_cost  # noqa: E402
from macro_place.utils import validate_placement  # noqa: E402

from hrt_placer.engine import AnalyticalPlacer  # noqa: E402
from placer import _PORTFOLIO  # noqa: E402

BASE = "external/MacroPlacement/Testcases/ICCAD04"


def main():
    names = sys.argv[1:] or ["ibm01", "ibm03", "ibm09", "ibm13", "ibm17"]
    print(f"{'bench':>7} {'config':>16} {'proxy':>8} {'wl':>7} {'den':>7} "
          f"{'cong':>8} {'valid':>6} {'t(s)':>7}")
    by_config = {cfg["name"]: [] for cfg in _PORTFOLIO}
    for name in names:
        bm, plc = load_benchmark_from_dir(f"{BASE}/{name}")
        for cfg in _PORTFOLIO:
            kwargs = {k: v for k, v in cfg.items() if k != "name"}
            placer = AnalyticalPlacer(device="cuda", **kwargs)
            t0 = time.time()
            try:
                placement = placer.place(bm)
            except Exception as exc:
                print(f"{name:>7} {cfg['name']:>16} CRASH {type(exc).__name__}: {exc}")
                continue
            dt = time.time() - t0
            try:
                costs = compute_proxy_cost(placement, bm, plc)
            except Exception as exc:
                print(f"{name:>7} {cfg['name']:>16} PROXY-CRASH "
                      f"{type(exc).__name__}: {exc}")
                continue
            valid, _ = validate_placement(placement, bm)
            proxy = costs["proxy_cost"]
            by_config[cfg["name"]].append(proxy)
            print(f"{name:>7} {cfg['name']:>16} {proxy:8.4f} "
                  f"{costs['wirelength_cost']:7.3f} "
                  f"{costs['density_cost']:7.3f} "
                  f"{costs['congestion_cost']:8.3f} "
                  f"{str(valid):>6} {dt:7.1f}")
    print("---")
    print(f"{'AVG':>7} {'config':>16} {'proxy':>8}")
    base_avg = None
    for cname, vals in by_config.items():
        if not vals:
            continue
        avg = sum(vals) / len(vals)
        if cname == "baseline":
            base_avg = avg
        tag = ""
        if base_avg is not None and cname != "baseline":
            tag = f"  ({(avg / base_avg - 1) * 100:+.2f}% vs baseline)"
        print(f"{'AVG':>7} {cname:>16} {avg:8.4f}{tag}")


if __name__ == "__main__":
    main()
