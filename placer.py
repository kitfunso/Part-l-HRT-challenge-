"""Submission entry point for the HRT/Partcl Macro Placement Challenge 2026.

The challenge evaluator imports this file and calls ``MyPlacer().place(benchmark)``.

``place`` runs the analytical engine, then refines with simulated annealing,
all under a hard wall-clock budget kept safely below the 1-hour-per-benchmark
cap. The engine always returns a legal placement, so a timeout at any stage
still yields a legal result.
"""

import os
import sys
import time
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
for _cand in (_HERE, os.path.join(_HERE, "src")):
    if _cand not in sys.path:
        sys.path.insert(0, _cand)

import torch  # noqa: E402

from hrt_placer.engine import AnalyticalPlacer  # noqa: E402
from hrt_placer.engine_v2 import AnalyticalPlacerV2  # noqa: E402
from hrt_placer.engine_v3 import AnalyticalPlacerV3  # noqa: E402
from hrt_placer.proxy_cost import ProxyCost  # noqa: E402
from hrt_placer.refine_sa import refine  # noqa: E402
from hrt_placer.refine_soft import refine_soft_macros  # noqa: E402
from hrt_placer.search import _is_legal  # noqa: E402


def _make_engine(cfg, device):
    """Instantiate the engine class named by ``cfg['engine']`` with the
    config's remaining kwargs. Defaults to AnalyticalPlacer (v1) when no
    ``engine`` key is present, so legacy single-candidate configs still work.
    """
    klass = {
        "v1": AnalyticalPlacer,
        "v2": AnalyticalPlacerV2,
        "v3": AnalyticalPlacerV3,
    }[cfg.get("engine", "v1")]
    kwargs = {k: v for k, v in cfg.items() if k not in ("name", "engine")}
    return klass(device=device, **kwargs)


# Engine portfolio. The placer runs every candidate, scores each via
# ``ProxyCost``, applies the soft-refine gate per candidate, and feeds the
# per-bench winner to SA. The per-bench ProxyCost gate makes adding a
# candidate strictly non-regressing: a candidate is only ever adopted on a
# benchmark where it scores lowest.
#
#   baseline -- AnalyticalPlacer (Adam, log-sum-exp HPWL, bin-overflow
#               density). Tuned defaults from the Step-3 TPE search.
#   edensity_300 / edensity_600 -- AnalyticalPlacerV3: electrostatic density
#               (FFT Poisson field over a bilinear-splatted density map)
#               instead of bin-overflow. The global repulsion spreads macros
#               where the local bin-overflow penalty leaves them clumped.
#               edensity_weight is benchmark-sensitive; 300 and 600 bracket
#               the optimum found on the 5-bench sweep (proxy is U-shaped in
#               the weight). Per-bench: v3 wins ibm17 (-2.5%) and ibm01,
#               v1 wins ibm13; portfolio-best AVG -1.73% vs v1-only.
#
# History (2026-05-20 - 2026-05-21):
#   - density_topk_frac=0.1 + congestion_weight=0.5: +3.5%, illegal ibm09.
#   - v2 (Nesterov + weighted-average WL): 0.00% net after the refine gate;
#     kept in tree (engine_v2.py) as a research artifact, not in the portfolio.
#   - eDensity first port rasterised macros as hard rectangles -> degenerate
#     gradient (+96%); fixed with bilinear density splatting (engine_v3.py).
_PORTFOLIO = [
    {"name": "baseline", "engine": "v1"},
    {"name": "edensity_300", "engine": "v3", "edensity_weight": 300.0},
    {"name": "edensity_600", "engine": "v3", "edensity_weight": 600.0},
]


class MyPlacer:
    """Hyperparameter-diverse engine portfolio + SA refinement under a hard budget."""

    # 55 min: margin under the 1-hour-per-benchmark evaluator cap.
    # ``HRT_TIME_BUDGET`` (seconds) overrides the default -- useful for
    # smoke tests and for judges who want a tighter wall-clock cap.
    def __init__(self, time_budget=3300.0):
        env = os.environ.get("HRT_TIME_BUDGET")
        if env:
            time_budget = float(env)
        # ``torch.cuda.is_available()`` returns True even when the bundled
        # kernels do not support the host GPU's compute capability (e.g.
        # pytorch 2.5.1 on Blackwell). A tiny smoke op confirms kernels
        # actually run; if it raises, fall back to CPU so the submission
        # still produces a legal placement on any host.
        self.device = "cpu"
        if torch.cuda.is_available():
            try:
                _ = (torch.zeros(2, device="cuda") + 1.0).sum().item()
                self.device = "cuda"
            except Exception as exc:
                print(f"[placer] CUDA available but kernel smoke failed "
                      f"({type(exc).__name__}: {exc}); falling back to CPU",
                      file=sys.stderr, flush=True)
        self.time_budget = time_budget

    def place(self, benchmark):
        t0 = time.time()

        # Stage 1 - hyperparameter-diverse engine portfolio. Total engine
        # budget is 45% of the wall-clock cap, split evenly across candidates.
        # Each candidate's deadline is absolute; the engine legalises (shelf-pack
        # fallback) even on a deadline hit, so every candidate's output is legal.
        # ProxyCost is constructed once and reused across candidates.
        n = len(_PORTFOLIO)
        per_candidate = 0.45 / max(1, n) * self.time_budget
        try:
            pc = ProxyCost(benchmark)
        except Exception as exc:
            print(f"[placer] ProxyCost construction failed "
                  f"({type(exc).__name__}: {exc}); using single baseline engine",
                  file=sys.stderr, flush=True)
            traceback.print_exc(file=sys.stderr)
            pc = None

        best_placement = None
        best_cost = float("inf")
        best_name = None
        for i, cfg in enumerate(_PORTFOLIO):
            cand_deadline = t0 + (i + 1) * per_candidate
            if time.time() > cand_deadline - 5.0:
                # Not enough remaining budget for this candidate; skip cleanly.
                continue
            try:
                placement = _make_engine(cfg, self.device).place(
                    benchmark, deadline=cand_deadline)
            except Exception as exc:
                print(f"[placer] engine candidate '{cfg['name']}' raised "
                      f"{type(exc).__name__}: {exc}; skipping",
                      file=sys.stderr, flush=True)
                traceback.print_exc(file=sys.stderr)
                continue
            if not _is_legal(placement, benchmark):
                print(f"[placer] engine candidate '{cfg['name']}' produced "
                      f"illegal placement; skipping", file=sys.stderr, flush=True)
                continue
            if pc is None:
                # No proxy scorer; first legal result wins, then break.
                best_placement = placement
                best_name = cfg["name"]
                break
            try:
                cost = float(pc(placement)["proxy_cost"])
            except Exception as exc:
                print(f"[placer] proxy scoring failed for '{cfg['name']}' "
                      f"({type(exc).__name__}: {exc}); skipping",
                      file=sys.stderr, flush=True)
                traceback.print_exc(file=sys.stderr)
                continue
            print(f"[placer] candidate '{cfg['name']}' proxy={cost:.4f}",
                  file=sys.stderr, flush=True)

            # Post-engine soft-macro refinement (Adam on soft slice, hard frozen)
            # with a per-candidate ProxyCost gate -- adopt the refined placement
            # only if it is strictly lower-proxy AND legal. Guaranteed
            # non-regression: the engine output is the safe fallback.
            refined_cost = cost
            try:
                refined_um, refine_dt = refine_soft_macros(
                    benchmark, placement, device=self.device)
                if _is_legal(refined_um, benchmark):
                    r_cost = float(pc(refined_um)["proxy_cost"])
                    print(f"[placer]   soft-refine '{cfg['name']}' "
                          f"proxy={r_cost:.4f} ({(r_cost/cost-1)*100:+.2f}%) "
                          f"dt={refine_dt:.1f}s",
                          file=sys.stderr, flush=True)
                    if r_cost < cost:
                        placement = refined_um
                        refined_cost = r_cost
                else:
                    print(f"[placer]   soft-refine '{cfg['name']}' produced "
                          f"illegal placement; keeping engine result",
                          file=sys.stderr, flush=True)
            except Exception as exc:
                print(f"[placer]   soft-refine '{cfg['name']}' raised "
                      f"{type(exc).__name__}: {exc}; keeping engine result",
                      file=sys.stderr, flush=True)
                traceback.print_exc(file=sys.stderr)

            if refined_cost < best_cost:
                best_cost = refined_cost
                best_placement = placement
                best_name = cfg["name"]

        # Last-resort fallback: no candidate finished legally within budget.
        # Re-run the baseline against the remaining engine budget; if even
        # that fails (e.g. CUDA driver/kernel mismatch on the eval host),
        # drop to CPU and retry. The engine's shelf-pack legaliser guarantees
        # a legal result on CPU, so this is the absolute submission safety net.
        if best_placement is None:
            print("[placer] portfolio produced no legal candidate; running "
                  "fallback baseline engine", file=sys.stderr, flush=True)
            fallback_deadline = t0 + 0.45 * self.time_budget
            try:
                best_placement = AnalyticalPlacer(device=self.device).place(
                    benchmark, deadline=fallback_deadline)
                best_name = "fallback_baseline"
            except Exception as exc:
                print(f"[placer] fallback engine on {self.device} raised "
                      f"{type(exc).__name__}: {exc}; retrying on CPU",
                      file=sys.stderr, flush=True)
                traceback.print_exc(file=sys.stderr)
                best_placement = AnalyticalPlacer(device="cpu").place(
                    benchmark, deadline=fallback_deadline)
                best_name = "fallback_baseline_cpu"

        # Fill best_cost when we skipped scoring (pc=None or pc broke). Keeps
        # the winner-log line meaningful instead of printing proxy=inf.
        if best_cost == float("inf") and pc is not None:
            try:
                best_cost = float(pc(best_placement)["proxy_cost"])
            except Exception:
                pass
        cost_str = f"{best_cost:.4f}" if best_cost != float("inf") else "n/a"
        print(f"[placer] portfolio winner='{best_name}' proxy={cost_str} "
              f"engine_elapsed={time.time() - t0:.1f}s",
              file=sys.stderr, flush=True)
        placement = best_placement

        # Stage 2 - SA refinement on the remaining budget. The engine result
        # is the safe fallback: only adopt the refined placement if it is
        # legal and refinement did not raise. Log any failure to stderr so
        # ProcessPool pickling errors, proxy crashes, or OOM are visible
        # in evaluator logs instead of silently removing Stage-2 gain.
        remaining = self.time_budget - (time.time() - t0)
        if remaining > 120.0:
            try:
                refined = refine(benchmark, placement,
                                 time_budget=remaining - 90.0)
                if _is_legal(refined, benchmark):
                    placement = refined
                else:
                    print("[placer] SA refinement produced illegal placement; "
                          "falling back to engine result", file=sys.stderr,
                          flush=True)
            except Exception as exc:
                print(f"[placer] SA refinement raised {type(exc).__name__}: "
                      f"{exc}; falling back to engine result",
                      file=sys.stderr, flush=True)
                traceback.print_exc(file=sys.stderr)

        return placement
