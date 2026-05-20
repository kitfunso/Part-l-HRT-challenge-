# Architecture — HRT/Partcl Macro Placement Challenge 2026

## Tech stack
- Python 3.11, PyTorch 2.5.1, CUDA 12.4 (base image `pytorch/pytorch:2.5.1-cuda12.4`).
- Portable PyTorch electrostatic analytical engine (our own code; CPU/GPU, no CUDA build).
- **Optuna** (MIT) — multi-objective TPE (MOTPE) hyperparameter search.
- **WireMask-BBO / WireMask-EA** (MIT) — HPWL refinement (pure Python, no build).
- numpy; the challenge harness for evaluation.
- Target hardware: AMD EPYC 9655P (16 cores, 100 GB) + NVIDIA RTX 6000 Ada (48 GB).

## Folder structure
```
.
├── placer.py                  # submission entry point: MyPlacer.place(benchmark)
├── src/hrt_placer/
│   ├── engine.py              # portable PyTorch electrostatic analytical placer
│   ├── proxy_cost.py          # exact TILOS proxy cost (WL + density + congestion)
│   ├── search.py              # TPE search over engine hyperparameters (dev-time)
│   ├── refine_sa.py           # hotspot-targeted SA + IncrementalProxy + 16-chain parallel
│   ├── timing.py              # topology-derived net criticality weighting (opt-in, currently unused in submission path)
│   └── select.py              # feasibility-gate-aware candidate selection (opt-in, currently unused in submission path)
├── scripts/
│   ├── calibrate.py           # calibrate ProxyCost vs real evaluator
│   └── dev_eval.py            # convenience harness for IBM regression runs
├── docs/                      # PRD, ARCHITECTURE, AI_RULES, PLAN
├── tests/                     # legality + timeout + determinism checks (Step 8)
└── requirements.txt
```

**Not included in the repo** (provided by the local dev workflow, not the
submission package):
- `external/macro-place-challenge-2026/` — challenge harness, cloned locally for
  evaluator runs (`uv run python -m macro_place.evaluate`). The judges drop their
  own harness around the submission, so it is not checked in.
- `Dockerfile` — not shipped. The judges' standard image
  (`pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime`, Python 3.11) provides every
  runtime dependency listed in `requirements.txt`; see `README.md` for the
  run-eval invocation.

## Data model
- `Benchmark`: macro sizes, initial positions, fixed mask, canvas/grid dims,
  net connectivity (geometry + netlist only — no timing/SDC).
- `PlacementCost`: the challenge's proxy-cost object (wraps TILOS plc client).
- Internal `Candidate`: macro coords + orientations + cached proxy-cost terms +
  estimated timing-safety score.

## Modules & responsibilities
| Module | Responsibility |
|--------|----------------|
| `placer.py` | Submission entry point; orchestrates engine + SA refinement under a hard wall-clock budget; logs SA failures to stderr |
| `engine.py` | Portable PyTorch electrostatic global placement; smooth HPWL + density-overflow + macro-overlap objective; CUDA-determinism pinned (cudnn deterministic, TF32 off) |
| `proxy_cost.py` | Compute the exact TILOS proxy cost (WL + density + congestion + macro-routing-blockage); expose grid routing primitives reused by SA |
| `search.py` | Self-contained TPE search over engine hyperparameters; dev-time tuner |
| `refine_sa.py` | Hotspot-targeted SA with `IncrementalProxy` (mutable proxy bookkeeping); 16-chain `ProcessPoolExecutor` parallel; best legal wins |
| `timing.py` | Net-weight model from BFS depth from I/O ports (opt-in; not invoked from `placer.py`) |
| `select.py` | Proxy-budget-gated candidate picker (uniform vs criticality-weighted) (opt-in; not invoked from `placer.py`) |

## Data flow (submission path)
`Benchmark` → `AnalyticalPlacer.place` (engine, deadline = 45% of budget)
→ `refine_sa.refine` (16 parallel SA chains on the engine's legal output,
budget = remaining minus a 90 s safety margin) → legal macro coords back to
the evaluator. The engine result is the safe fallback at every stage.

## External dependencies
- Runtime: only `torch>=2.5,<2.6` and `numpy>=1.26,<3.0` (see `requirements.txt`),
  both provided by the judges' standard image. No pip installs are needed at
  evaluation time, which is required because the evaluator runs with `--network none`.
