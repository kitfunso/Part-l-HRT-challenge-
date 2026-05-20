"""Candidate-placement selection with a proxy-cost feasibility gate.

Runs the analytical engine twice on a benchmark -- once with uniform net
weights (Tier-1 baseline) and once with topology-derived criticality
weights (Tier-2 lean) -- then picks the candidate that minimises a
weighted-wirelength surrogate subject to a hard Tier-1 proxy budget.

The proxy budget is the safety valve: a timing-weighted placement is
only adopted when its unweighted proxy is within ``proxy_budget`` of
the best candidate's proxy. This honours CLAUDE.md's rule that "a
lower proxy cost that worsens estimated timing is NOT an improvement"
in the reverse direction -- we will not trade away Tier-1 ranking for
a speculative Tier-2 gain.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch

from .engine import AnalyticalPlacer
from .proxy_cost import ProxyCost
from .timing import criticality_weighted_wirelength, net_criticality_weights


@dataclass
class Candidate:
    name: str
    placement: torch.Tensor
    proxy: float
    wirelength: float
    density: float
    congestion: float
    cwl: float
    runtime: float = 0.0


@dataclass
class SelectionResult:
    pick: Candidate
    reason: str
    candidates: list[Candidate] = field(default_factory=list)


def _score(name, placement, benchmark, crit_weights, runtime):
    # The evaluator scores proxy with the benchmark's default net weights,
    # ignoring whatever we drove the engine with; reset before ProxyCost.
    saved = benchmark.net_weights
    benchmark.net_weights = torch.ones_like(saved)
    try:
        pc = ProxyCost(benchmark)(placement)
    finally:
        benchmark.net_weights = saved
    return Candidate(
        name=name,
        placement=placement,
        proxy=float(pc["proxy_cost"]),
        wirelength=float(pc["wirelength_cost"]),
        density=float(pc["density_cost"]),
        congestion=float(pc["congestion_cost"]),
        cwl=criticality_weighted_wirelength(placement, benchmark, crit_weights),
        runtime=runtime,
    )


def select_placement(
    benchmark,
    proxy_budget: float = 0.01,
    timing_alpha: float = 1.0,
    placer_factory=None,
) -> SelectionResult:
    """Generate baseline + timing-weighted placements and pick by gated score.

    ``proxy_budget`` -- the timing-weighted candidate is adopted only when
    its unweighted proxy is within (1 + proxy_budget) of the baseline's.
    The default 0.01 (1%) is intentionally conservative because the cwl
    score is an unvalidated surrogate (no OpenROAD timer in the loop).
    ``timing_alpha`` -- depth-criticality weight scale; nets at max depth
    get weight (1 + alpha).
    """
    if placer_factory is None:
        placer_factory = AnalyticalPlacer

    crit = net_criticality_weights(benchmark, alpha=timing_alpha)
    orig_weights = benchmark.net_weights.clone()

    benchmark.net_weights = torch.ones_like(orig_weights)
    t0 = time.time()
    pl_uniform = placer_factory().place(benchmark)
    uniform = _score("uniform", pl_uniform, benchmark, crit, time.time() - t0)

    benchmark.net_weights = crit.clone()
    t0 = time.time()
    pl_weighted = placer_factory().place(benchmark)
    weighted = _score(
        "crit_weighted", pl_weighted, benchmark, crit, time.time() - t0)

    benchmark.net_weights = orig_weights

    best_proxy = min(uniform.proxy, weighted.proxy)
    proxy_cap = best_proxy * (1.0 + proxy_budget)
    eligible = [c for c in (uniform, weighted) if c.proxy <= proxy_cap]
    pick = min(eligible, key=lambda c: c.cwl)

    if pick is uniform:
        reason = (f"baseline kept: weighted proxy "
                  f"{weighted.proxy:.4f} vs cap {proxy_cap:.4f}"
                  if weighted.proxy > proxy_cap
                  else f"baseline cwl {uniform.cwl:.4f} <= "
                       f"weighted cwl {weighted.cwl:.4f}")
    else:
        reason = (f"crit-weighted adopted: cwl "
                  f"{weighted.cwl:.4f} < {uniform.cwl:.4f}, "
                  f"proxy {weighted.proxy:.4f} <= cap {proxy_cap:.4f}")

    return SelectionResult(pick=pick, reason=reason,
                           candidates=[uniform, weighted])
