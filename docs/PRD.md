# Product Requirements Document — HRT/Partcl Macro Placement Challenge 2026

## Summary
A macro placer for the Partcl/HRT Macro Placement Challenge 2026. It exposes a
`placer.py` implementing the challenge interface and is built to win the
**$20,000 Grand Prize** — judged by real OpenROAD place-and-route timing
(WNS/TNS/Area) on NG45 designs — by first clearing the top-7 proxy-cost gate.

## Problem
Place hard macros (SRAMs, IP) and movable soft macros on chip canvases with
zero hard-macro overlap, minimizing the TILOS proxy cost
`1.0·Wirelength + 0.5·Density + 0.5·Congestion` across 17 IBM ICCAD04
benchmarks, within 1 hour per benchmark on one RTX 6000 Ada GPU. The Grand
Prize is then decided by real PnR timing, where the proxy cost is only a
weak correlate — so the placement must also be genuinely routable and
timing-safe.

## Target users
The challenge evaluator and judges. The deliverable is an automated placer,
not an interactive tool.

## What it IS — in scope (reflects what shipped)
- A `placer.py` implementing the challenge's `place(benchmark) -> coords` API,
  with stderr logging on SA failure so the eval host shows real errors instead
  of a silent Stage-2 dropout.
- A **portable PyTorch electrostatic** analytical engine (ePlace/RePlAce-style)
  with smooth HPWL + bin-density overflow + hard-macro-overlap loss, runs
  identically on CPU and GPU, no CUDA build toolchain. CUDA-determinism pinned
  (cudnn deterministic, TF32 off, allow_tf32 off) so reruns are byte-identical.
- A **self-contained TPE search loop** (no Optuna dependency) over engine
  hyperparameters, dev-time tuner; tuned defaults baked into the engine.
- **Hotspot-targeted simulated annealing** with `IncrementalProxy` for O(moved
  net) cost updates; 16 parallel chains via `ProcessPoolExecutor`; best legal
  wins. Always-legal-by-construction: every accepted move stays legal.
- **>=12 um macro-to-macro clearance** in submitted placements (clamped on
  NG45 canvases; IBM abstract-unit dies use a smaller proportional clearance)
  so Tier 2's auto-spacing pass does not override our coordinates.
- Hard internal per-benchmark wall-clock budget (default 3300 s, env-overridable
  via `HRT_TIME_BUDGET`) that returns the best legal placement found so far.

## What it is NOT — out of scope (descoped during build)
- **NOT reinforcement learning** — no Circuit Training / MaskPlace / ChiPFormer
  / EfficientPlace. "Stronger Baselines" + BBOPlace-Bench show black-box
  optimization beats RL without training cost.
- **NOT dependent on a CUDA build toolchain** — the analytical engine is plain
  PyTorch, testable on CPU and portable to the eval GPU with no compilation.
- **NOT a heavyweight standard-cell placer** — only a lean macro-focused
  analytical core, not a full DREAMPlace-scale flow.
- **NOT proxy-overfitting** — a lower proxy score is worthless if it fails the
  Tier 2 timing gate.
- **NOT** benchmark-specific hardcoding, evaluation-function edits, 90°
  rotations, soft-macro resizing, or proprietary tools (all forbidden by rules).
- **NOT** sequence-pair / B*-tree compaction representations (wrong fit for
  43–53% utilization).
- **NOT** an interactive or general-purpose placement tool.
- **NOT** a WireMask-EA HPWL refinement pass — descoped in Step 5: HPWL is only
  ~5% of the proxy on our placements; WireMask cannot move the dominant
  density+congestion terms enough to close the gap.
- **NOT** Klein-4 orientation local search — descoped in Step 5: the challenge
  `place(benchmark)` API returns positions only, the proxy applies fixed
  benchmark pin offsets, and the DEF writer emits each node's default
  orientation. There is no submission channel for orientations.
- **NOT** GPU soft-macro co-optimization — descoped: not on the critical path
  to either the Tier-1 proxy gate or the Grand Prize timing gate within the
  build budget.
- **NOT** topology-derived timing-criticality net weighting in the submission
  path — `src/hrt_placer/timing.py` and `src/hrt_placer/select.py` are
  implemented but unused. Codex review (2026-05-20) confirmed the 1% proxy
  budget gate inside `select.py` is structurally incapable of closing the
  Tier-1 gap (gap ~25%, gate ~1%). Kept in the tree as future work pending an
  OpenROAD-in-the-loop timer.
- **NOT** a separate `portfolio.py` runner — `placer.py` already does engine
  budget allocation + SA timeout; a separate module would duplicate that.
- **NOT** a shipped `Dockerfile` — the judges' standard image
  (`pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime`, Python 3.11) provides every
  runtime dependency in `requirements.txt`. The submission keeps zero pip
  installs at eval time so `--network none` is satisfied.

## Core flows
1. Evaluator calls `MyPlacer.place(benchmark)`; placer returns macro center
   coordinates as a tensor.
2. Internal pipeline: `AnalyticalPlacer.place` (engine, 45% of the per-benchmark
   wall-clock budget, deterministic CUDA) -> if a legal SA refinement fits in
   the remaining budget, run 16 parallel SA chains and adopt the legal best
   -> return. Engine result is the safe fallback if SA fails or runs short.

## Success criteria
- Top-7 proxy-cost ranking (target proxy ~<=1.05 across the 17 benchmarks).
- Beats both SA and RePlAce baselines on NG45 WNS/TNS/Area in OpenROAD PnR.
- Every returned placement is legal: zero hard-macro overlap, in-canvas,
  fixed macros pinned.
- Completes within the 1-hour-per-benchmark cap on the target hardware.
- Fully reproducible from the provided Docker image with no network access.
