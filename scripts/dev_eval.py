"""Development evaluation harness.

Runs our placer on the IBM benchmarks and prints proxy cost vs the published
SA / RePlAce baselines. Run inside the challenge venv, e.g.:

    cd external/macro-place-challenge-2026
    uv run python ../../scripts/dev_eval.py ibm01 ibm02 ibm03 ibm04 ibm06
"""

import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHALLENGE = os.path.join(REPO, "external", "macro-place-challenge-2026")
sys.path.insert(0, os.path.join(REPO, "src"))
os.chdir(CHALLENGE)

from macro_place.loader import load_benchmark_from_dir  # noqa: E402
from macro_place.objective import compute_proxy_cost  # noqa: E402
from macro_place.utils import validate_placement  # noqa: E402

from hrt_placer.engine import AnalyticalPlacer  # noqa: E402

BASE = "external/MacroPlacement/Testcases/ICCAD04"

# Published baselines (README comparison table).
SA = {"ibm01": 1.3166, "ibm02": 1.9072, "ibm03": 1.7401, "ibm04": 1.5037,
      "ibm06": 2.5057, "ibm07": 2.0229, "ibm08": 1.9239, "ibm09": 1.3875,
      "ibm10": 2.1108, "ibm11": 1.7111, "ibm12": 2.8261, "ibm13": 1.9141,
      "ibm14": 2.2750, "ibm15": 2.3000, "ibm16": 2.2337, "ibm17": 3.6726,
      "ibm18": 2.7755}
RP = {"ibm01": 0.9976, "ibm02": 1.8370, "ibm03": 1.3222, "ibm04": 1.3024,
      "ibm06": 1.6187, "ibm07": 1.4633, "ibm08": 1.4285, "ibm09": 1.1194,
      "ibm10": 1.5009, "ibm11": 1.1774, "ibm12": 1.7261, "ibm13": 1.3355,
      "ibm14": 1.5436, "ibm15": 1.5159, "ibm16": 1.4780, "ibm17": 1.6446,
      "ibm18": 1.7722}


def main():
    names = sys.argv[1:] or ["ibm01", "ibm02", "ibm03", "ibm04", "ibm06"]
    print(f"{'bench':>7} {'proxy':>8} {'wl':>7} {'den':>7} {'cong':>8} "
          f"{'SA':>8} {'RP':>8} {'vsSA':>7} {'vsRP':>7} {'valid':>6} {'t(s)':>7}")
    rows = []
    for name in names:
        bm, plc = load_benchmark_from_dir(f"{BASE}/{name}")
        placer = AnalyticalPlacer(verbose=True)
        t0 = time.time()
        placement = placer.place(bm)
        dt = time.time() - t0
        costs = compute_proxy_cost(placement, bm, plc)
        valid, viol = validate_placement(placement, bm)
        proxy = costs["proxy_cost"]
        sa, rp = SA.get(name), RP.get(name)
        vs_sa = f"{(proxy / sa - 1) * 100:+.1f}%" if sa else "-"
        vs_rp = f"{(proxy / rp - 1) * 100:+.1f}%" if rp else "-"
        rows.append((name, proxy, sa, rp, valid))
        print(f"{name:>7} {proxy:8.4f} {costs['wirelength_cost']:7.3f} "
              f"{costs['density_cost']:7.3f} {costs['congestion_cost']:8.3f} "
              f"{(sa or 0):8.4f} {(rp or 0):8.4f} {vs_sa:>7} {vs_rp:>7} "
              f"{str(valid):>6} {dt:7.2f}")
        if not valid:
            print(f"        violations: {viol}")
    if rows:
        avg = sum(r[1] for r in rows) / len(rows)
        print(f"{'AVG':>7} {avg:8.4f}")


if __name__ == "__main__":
    main()
