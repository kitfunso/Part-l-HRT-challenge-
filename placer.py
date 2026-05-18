"""Submission entry point for the HRT/Partcl Macro Placement Challenge 2026.

The challenge evaluator imports this file and calls ``MyPlacer().place(benchmark)``.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
for _cand in (_HERE, os.path.join(_HERE, "src")):
    if _cand not in sys.path:
        sys.path.insert(0, _cand)

import torch  # noqa: E402

from hrt_placer.engine import AnalyticalPlacer  # noqa: E402


class MyPlacer:
    """Analytical macro placer wrapper."""

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def place(self, benchmark):
        placer = AnalyticalPlacer(device=self.device)
        return placer.place(benchmark)
