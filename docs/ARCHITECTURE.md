# Architecture — HRT/Partcl Macro Placement Challenge 2026

## Tech stack
- Python 3.11, PyTorch 2.5.1, CUDA 12.4 (base image `pytorch/pytorch:2.5.1-cuda12.4`).
- **DREAMPlace** (BSD-3) — GPU analytical global-placement engine.
- **WireMask-BBO / WireMask-EA** (MIT) — HPWL refinement.
- **AutoDMP** (Apache-2.0) — MOTPE multi-objective search loop (extracted, not forked whole).
- Optuna (or AutoDMP's bundled BO) for the Bayesian search.
- numpy; the challenge harness for evaluation.
- Target hardware: AMD EPYC 9655P (16 cores, 100 GB) + NVIDIA RTX 6000 Ada (48 GB).

## Folder structure
```
.
├── placer.py                  # submission entry point: MyPlacer.place(benchmark)
├── src/hrt_placer/
│   ├── engine.py              # DREAMPlace global-placement wrapper
│   ├── proxy_cost.py          # exact TILOS proxy cost (WL + density + congestion)
│   ├── search.py              # MOTPE/Bayesian loop over engine hyperparameters
│   ├── refine_sa.py           # hotspot-targeted SA, incremental cost
│   ├── refine_wiremask.py     # WireMask-EA HPWL pass
│   ├── orientation.py         # Klein-4 orientation local search
│   ├── timing.py              # topology-derived net criticality weighting
│   ├── select.py              # feasibility-gate-aware candidate selection
│   └── portfolio.py           # per-benchmark runner + timeout handling
├── external/
│   ├── DREAMPlace/            # submodule (BSD-3)
│   ├── WireMask-BBO/          # submodule (MIT)
│   └── macro-place-challenge-2026/  # challenge harness, reference only
├── docs/                      # PRD, ARCHITECTURE, AI_RULES, PLAN
├── tests/                     # legality + timeout + per-benchmark checks
├── Dockerfile
└── requirements.txt
```

## Data model
- `Benchmark`: macro sizes, initial positions, fixed mask, canvas/grid dims,
  net connectivity (geometry + netlist only — no timing/SDC).
- `PlacementCost`: the challenge's proxy-cost object (wraps TILOS plc client).
- Internal `Candidate`: macro coords + orientations + cached proxy-cost terms +
  estimated timing-safety score.

## Modules & responsibilities
| Module | Responsibility |
|--------|----------------|
| `engine.py` | Run DREAMPlace global placement for a given hyperparameter set |
| `proxy_cost.py` | Compute the exact TILOS proxy cost; expose incremental updates |
| `search.py` | MOTPE search over hyperparameters; objective = exact proxy cost |
| `refine_sa.py` | Hotspot SA on top-5%/10% bins; incremental delta evaluation |
| `refine_wiremask.py` | WireMask-EA wire-mask refinement of HPWL |
| `orientation.py` | Per-macro best-of-4 Klein-4 orientation search |
| `timing.py` | Net-weight model from logic depth / fanout-cone topology |
| `select.py` | Choose proxy-good + timing-safe candidate; enforce feasibility gate |
| `portfolio.py` | Budget allocation across stages; hard 1-hour timeout per benchmark |

## Data flow
`Benchmark` -> `engine` (global place) -> `search` (BO over configs, scored by
`proxy_cost`) -> Pareto candidates -> `refine_sa` + `refine_wiremask` +
`orientation` -> `select` (proxy + timing-safety) -> legal macro coords back to
the evaluator. `timing.py` supplies net weights consumed by `engine` and
`refine_*`. `portfolio.py` wraps the whole flow with the time budget.

## External dependencies
- DREAMPlace, WireMask-BBO, AutoDMP: pinned git submodules, vendored licenses.
- All dependencies bundled into the Docker image — the evaluator runs with
  `--network none`, so no runtime downloads are permitted.
