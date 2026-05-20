# hrt-placer

Submission for the [Partcl / HRT Macro Placement Challenge 2026][challenge].

[challenge]: https://github.com/partcleda/partcl-macro-place-challenge

## Approach

Two-stage macro placer, both stages always produce a legal placement so a
wall-clock timeout at any point still yields a submittable result:

1. **Analytical engine** (`src/hrt_placer/engine.py`) — gradient descent on
   a smoothed HPWL + density + overlap objective in plain PyTorch (no
   CUDA toolchain required at build time). Closes with a deterministic
   shelf-pack legaliser and a push-apart pass that targets the
   `SCORING.md` ≥ 12 µm clearance on NG45-scale dies.

2. **Simulated-annealing refinement** (`src/hrt_placer/refine_sa.py`) —
   hotspot-targeted hard-macro moves over the incremental proxy cost
   (re-routes only the moved macro's nets each step). Multiple chains
   run on separate processes and the best legal result wins.

Optional topology-driven net-criticality weighting
(`src/hrt_placer/timing.py`) and a feasibility-gated candidate selector
(`src/hrt_placer/select.py`) are included as opt-in tools; the
submission path keeps the proxy-validated uniform-weight baseline.

## Running

The judges' standard image (`pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime`,
Python 3.11) has every runtime dependency, so no Dockerfile is shipped.
The repository directory is mounted into the evaluator and `placer.py`
adds `./src` to `sys.path` itself.

```bash
# From inside external/macro-place-challenge-2026:
uv run python -m macro_place.evaluate ../../placer.py -b ibm01
uv run python -m macro_place.evaluate ../../placer.py --all          # 17 IBM
uv run python -m macro_place.evaluate ../../placer.py --ng45         # 4 NG45

# Air-gapped Docker run via the challenge's run_eval.sh:
./external/macro-place-challenge-2026/eval_docker/run_eval.sh \
    hrt-placer placer.py
```

`HRT_TIME_BUDGET=<seconds>` overrides the default 3300 s per-benchmark cap;
useful for short smoke runs.

## Repository layout

```
placer.py                   Submission entry point (defines MyPlacer)
src/hrt_placer/
  engine.py                 Analytical placer + legaliser
  refine_sa.py              Simulated-annealing refinement
  proxy_cost.py             In-house faithful proxy-cost reimplementation
  search.py                 TPE hyperparameter search (dev-time)
  timing.py                 Topology-derived net-criticality weights
  select.py                 Proxy-gated candidate selection
scripts/dev_eval.py         Convenience harness for IBM regression runs
docs/                       PRD, architecture notes, rolling plan
```

## Dependencies

Listed in `requirements.txt`. The runtime needs only `torch` and `numpy`;
everything else used at submission time is Python stdlib.

## License

Apache 2.0 — see `LICENSE`.
