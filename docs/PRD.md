# Product Requirements Document — HRT/Partcl Macro Placement Challenge 2026

## Summary
A macro placer for the Partcl/HRT Macro Placement Challenge 2026. It exposes a
`placer.py` implementing the challenge interface and is built to win the
**$20,000 Grand Prize** — judged by real OpenROAD place-and-route timing
(WNS/TNS/Area) on NG45 designs — by first clearing the top-7 proxy-cost gate.

## Problem
Place hard macros (SRAMs, IP) and movable soft macros on chip canvases with
zero hard-macro overlap, minimizing the TILOS proxy cost
`1.0·Wirelength + 0.5·Density + 0.5·Congestion` across 18 IBM ICCAD04
benchmarks, within 1 hour per benchmark on one RTX 6000 Ada GPU. The Grand
Prize is then decided by real PnR timing, where the proxy cost is only a
weak correlate — so the placement must also be genuinely routable and
timing-safe.

## Target users
The challenge evaluator and judges. The deliverable is an automated placer,
not an interactive tool.

## What it IS — in scope
- A `placer.py` implementing the challenge's `place(benchmark) -> coords` API.
- A **DREAMPlace** (GPU analytical) global-placement engine.
- A **Bayesian/MOTPE search loop** (AutoDMP recipe) whose objective is the
  *exact TILOS proxy cost*, not DREAMPlace's generic metrics.
- **Hotspot-targeted simulated annealing** that refines the top-5%/top-10%
  bins the cost actually measures, with incremental cost evaluation.
- A **WireMask-EA** HPWL refinement pass.
- **Klein-4 orientation local search** (N/FN/FS/S).
- **Topology-derived timing-criticality net weighting** (logic depth, fanout
  cones) — there is no SDC in the placer interface.
- **Feasibility-gate-aware candidate selection**: never ship a placement
  expected to fail Tier 2's `WNS_sub >= min(WNS_SA, WNS_RP)` gate.
- A portfolio runner with strict per-benchmark timeout handling.
- A reproducible Dockerfile (`pytorch/pytorch:2.5.1-cuda12.4`, Python 3.11).

## What it is NOT — out of scope
- **NOT reinforcement learning** — no Circuit Training / MaskPlace / ChiPFormer
  / EfficientPlace. "Stronger Baselines" + BBOPlace-Bench show black-box
  optimization beats RL without training cost.
- **NOT a full fork of the AutoDMP pipeline** — extract only the MOTPE search
  loop and DREAMPlace glue.
- **NOT a custom analytical placer written from scratch** — stand on DREAMPlace.
- **NOT proxy-overfitting** — a lower proxy score is worthless if it fails the
  Tier 2 timing gate.
- **NOT** benchmark-specific hardcoding, evaluation-function edits, 90°
  rotations, soft-macro resizing, or proprietary tools (all forbidden by rules).
- **NOT** sequence-pair / B*-tree compaction representations (wrong fit for
  43–53% utilization).
- **NOT** an interactive or general-purpose placement tool.

## Core flows
1. Evaluator calls `MyPlacer.place(benchmark)`; placer returns macro center
   coordinates as a tensor.
2. Internal pipeline: load benchmark -> DREAMPlace global placement ->
   MOTPE search over engine hyperparameters scored by the exact proxy cost ->
   hotspot SA refinement -> WireMask-EA HPWL pass -> Klein-4 orientation
   search -> feasibility-gate-aware selection of the best candidate -> return.

## Success criteria
- Top-7 proxy-cost ranking (target proxy ~<=1.05 across the 18 benchmarks).
- Beats both SA and RePlAce baselines on NG45 WNS/TNS/Area in OpenROAD PnR.
- Every returned placement is legal: zero hard-macro overlap, in-canvas,
  fixed macros pinned.
- Completes within the 1-hour-per-benchmark cap on the target hardware.
- Fully reproducible from the provided Docker image with no network access.
