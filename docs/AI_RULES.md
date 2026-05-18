# AI Rules — HRT/Partcl Macro Placement Challenge 2026

The AI must follow every rule here on every change.

## Stack constraints
- Python 3.11 + PyTorch 2.5.1 + CUDA 12.4 only; target the
  `pytorch/pytorch:2.5.1-cuda12.4` base image.
- No new dependency without checking its license: **Apache-2.0, MIT, BSD, or
  GPL only** — the winning submission must be open-source-compatible.
- GPU-first: the RTX 6000 Ada (48 GB) is the compute budget; design for one GPU.

## Code organization
- Files go where `docs/ARCHITECTURE.md` says; `placer.py` stays a thin entry
  point delegating to `src/hrt_placer/`.
- Third-party code lives under `external/` as pinned submodules — never
  copy-paste it into `src/`.
- Keep each module single-responsibility per the architecture table.

## Security & runtime
- The evaluator runs with `--network none`: **no network calls at placement
  time**. All models, data, and dependencies must be baked into the image.
- No secrets, credentials, or API keys in the repo or image.

## Quality bar
- Every returned placement MUST be legal: zero hard-macro overlap, all macros
  in-canvas, fixed macros unmoved, only Klein-4 orientations.
- The placer MUST finish within 1 hour per benchmark — implement a hard
  internal timeout that returns the best legal placement found so far.
- Validate on all 18 IBM benchmarks before declaring any step done.
- A change that lowers proxy cost but worsens estimated timing is NOT an
  improvement — the Tier 2 gate `WNS_sub >= min(WNS_SA, WNS_RP)` governs.

## Forbidden
- Reinforcement learning approaches (training cost not justified).
- Modifying the challenge's evaluation/scoring functions.
- Benchmark-specific hardcoded placements or solutions.
- 90° rotations (R90/R270/FE/FW); resizing soft macros.
- Proprietary or commercial placement tools.
- Forking the entire AutoDMP pipeline — extract only the MOTPE loop.

## Workflow
- Follow `docs/PLAN.md` one step at a time; never skip ahead.
- New scope goes into `docs/PRD.md` (in scope or non-goals) before it is built.
