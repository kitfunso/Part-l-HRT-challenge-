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
from hrt_placer.proxy_cost import ProxyCost  # noqa: E402
from hrt_placer.refine_sa import refine  # noqa: E402
from hrt_placer.search import _is_legal  # noqa: E402


# Engine portfolio. The placer scores each candidate by ``ProxyCost`` and
# feeds the legal winner to SA. A short hyperparameter-diverse portfolio was
# trialled on 2026-05-20 (``proxy_density`` and ``congestion`` candidates with
# top-k density loss + a differentiable congestion surrogate) but regressed
# proxy by +3.5% on a 5-benchmark sample (ibm01/03/09/13/17) and produced an
# illegal placement on ibm09. Top-k density pushes macros apart more
# aggressively, raising routing demand faster than density falls -- net
# proxy worse. Reverted to the single tuned baseline (which is the same
# AnalyticalPlacer the prior submission used). The unused tuning knobs
# (``density_topk_frac``, ``congestion_weight``) remain on ``AnalyticalPlacer``
# for future TPE search but default to original behaviour.
_PORTFOLIO = [
    {"name": "baseline"},
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
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
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
            kwargs = {k: v for k, v in cfg.items() if k != "name"}
            try:
                placement = AnalyticalPlacer(device=self.device, **kwargs).place(
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
            if cost < best_cost:
                best_cost = cost
                best_placement = placement
                best_name = cfg["name"]

        # Last-resort fallback: no candidate finished legally within budget.
        # Re-run the baseline against the remaining engine budget (still bounded
        # below the 45% engine share). Should not fire in practice because the
        # engine's shelf-pack legaliser guarantees a legal result.
        if best_placement is None:
            print("[placer] portfolio produced no legal candidate; running "
                  "fallback baseline engine", file=sys.stderr, flush=True)
            fallback_deadline = t0 + 0.45 * self.time_budget
            best_placement = AnalyticalPlacer(device=self.device).place(
                benchmark, deadline=fallback_deadline)
            best_name = "fallback_baseline"

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
