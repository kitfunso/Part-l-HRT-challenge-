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

_HERE = os.path.dirname(os.path.abspath(__file__))
for _cand in (_HERE, os.path.join(_HERE, "src")):
    if _cand not in sys.path:
        sys.path.insert(0, _cand)

import torch  # noqa: E402

from hrt_placer.engine import AnalyticalPlacer  # noqa: E402
from hrt_placer.refine_sa import refine  # noqa: E402
from hrt_placer.search import _is_legal  # noqa: E402


class MyPlacer:
    """Analytical placement + SA refinement under a hard time budget."""

    # 55 min: margin under the 1-hour-per-benchmark evaluator cap.
    def __init__(self, time_budget=3300.0):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.time_budget = time_budget

    def place(self, benchmark):
        t0 = time.time()

        # Stage 1 — analytical engine. Gets up to 45% of the budget; the
        # engine legalizes even on a deadline hit, so this is always legal.
        engine_deadline = t0 + 0.45 * self.time_budget
        placement = AnalyticalPlacer(device=self.device).place(
            benchmark, deadline=engine_deadline)

        # Stage 2 — SA refinement on the remaining budget. The engine result
        # is the safe fallback: only adopt the refined placement if it is
        # legal and refinement did not raise.
        remaining = self.time_budget - (time.time() - t0)
        if remaining > 120.0:
            try:
                refined = refine(benchmark, placement,
                                 time_budget=remaining - 90.0)
                if _is_legal(refined, benchmark):
                    placement = refined
            except Exception:
                pass

        return placement
