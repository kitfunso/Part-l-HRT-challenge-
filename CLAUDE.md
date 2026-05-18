# CLAUDE.md

## Project
Solution for the Partcl/HRT Macro Placement Challenge 2026. Goal: the
$20,000 Grand Prize (real OpenROAD PnR timing on NG45). See `docs/PRD.md`,
`docs/ARCHITECTURE.md`, `docs/AI_RULES.md`, `docs/PLAN.md`.

## Non-negotiable rules
- Apache-2.0 / MIT / BSD / GPL dependencies only — the winning submission must
  be open-source-compatible.
- The evaluator runs with `--network none`: no network calls at placement
  time; bake everything into the Docker image.
- Every returned placement MUST be legal: zero hard-macro overlap, in-canvas,
  fixed macros unmoved, Klein-4 orientations only (N/FN/FS/S — no 90°).
- The placer MUST finish within 1 hour per benchmark; use a hard internal
  timeout that returns the best legal placement found so far.
- A lower proxy cost that worsens estimated timing is NOT an improvement —
  the Tier 2 gate `WNS_sub >= min(WNS_SA, WNS_RP)` governs.

## Forbidden
- Reinforcement learning approaches.
- Modifying the challenge evaluation/scoring functions.
- Benchmark-specific hardcoded solutions; 90° rotations; soft-macro resizing;
  proprietary placement tools.
- Forking the whole AutoDMP pipeline — extract only the MOTPE search loop.

## Workflow
- Follow `docs/PLAN.md` one step at a time; never skip ahead.
- New scope goes into `docs/PRD.md` before it gets built.
