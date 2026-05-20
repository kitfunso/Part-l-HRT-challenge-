# Build Plan — HRT/Partcl Macro Placement Challenge 2026

Deadline: **May 21, 2026, 11:59 PM Pacific**. Three days. Work ONE step at a
time: finish it, check it off, confirm with the user, then move to the next.

## How to use this plan
- Do the current step only — the first one whose checkbox is unchecked.
- A step is done when its "Done when" check passes.
- Mark the checkbox `[x]`, confirm with the user, then start the next step.

## Day 1 — foundation (critical path)

### [x] Step 1 — Stand up the evaluation harness
- Goal: clone the challenge repo into `external/`, install deps, get the
  evaluator running; reproduce the RePlAce (~1.46) and SA (~2.13) baselines.
- Files: `external/macro-place-challenge-2026/`, `requirements.txt`, `Dockerfile`.
- Done when: `evaluate` runs an example placer on ibm01 and prints a proxy cost
  matching the published baselines.
- DONE: challenge repo + TILOS submodule cloned, `uv sync` installed deps,
  evaluator runs all 17 benchmarks. Greedy placer reproduces the README table
  exactly (AVG 2.2109; SA 2.1251; RePlAce 1.4578). SA/RePlAce are published
  reference numbers from the TILOS paper, not runnable placers in the repo.

### [x] Step 2 — Portable PyTorch analytical engine + valid `placer.py`
- Goal: implement a plain-PyTorch electrostatic analytical placer (CPU/GPU, no
  CUDA build) and a `placer.py` returning a legal placement scored end-to-end.
- Files: `placer.py`, `src/hrt_placer/engine.py`.
- Done when: our placer produces legal placements and clearly beats the greedy
  (2.21) and SA (2.13) baselines on ibm01-ibm05; closing to RePlAce-level proxy
  is targeted via the exact-proxy objective in Step 3.
- DONE: smooth-HPWL + bin-density-overflow + overlap-barrier objective with
  Adam, push-apart legalization, shelf-pack fallback. All 17 benchmarks legal
  (zero overlap, in-canvas). AVG proxy 1.7748 vs SA 2.1251 (-16.5%, beats SA on
  16/17) and greedy 2.2109; RePlAce 1.4578 still ahead. Congestion is the
  dominant remaining term (~0.7 of proxy) and is unoptimized until Step 3.
  Outliers ibm15/ibm17 (high density) flagged for Step 3-5 refinement.

## Day 2 — the moat

### [x] Step 3 — Exact proxy cost as the search objective
- Goal: wire the exact TILOS proxy cost as a black-box objective; add the
  MOTPE/Bayesian search loop over DREAMPlace hyperparameters.
- Files: `src/hrt_placer/proxy_cost.py`, `src/hrt_placer/search.py`.
- Done when: BO search measurably lowers proxy cost vs stock DREAMPlace on a
  10-benchmark sample.
- DONE: `proxy_cost.py` is a faithful from-Benchmark port of plc_client_os
  (grid routing for congestion, exact rect/bin density, ABU-5) — calibrates
  to the real evaluator at Pearson ~0.98, density exact. `search.py` is a
  self-contained TPE optimizer. Widened-bound search over a 10-benchmark
  sample cut proxy -15.9% mean; the tuned config (near-minimal intervention
  — density spreading off, gentle WL+overlap cleanup of the initial
  placement) is now the engine default. Full 17-benchmark validation: AVG
  proxy 1.2971 (was 1.7748), legal on all 17, beats RePlAce (1.4578) on
  every benchmark (-5% to -27%, -11% mean) and SA on every benchmark — the
  Step 2 "RePlAce still ahead" gap is closed. CAVEAT: the tuned config
  assumes a usable initial placement; runtime per-design self-tuning via
  search.py is the Tier-2 hedge, to be wired up in Step 7.

### [x] Step 4 — Hotspot-targeted SA refinement
- Goal: SA that attacks the top-5%/top-10% bins, with incremental cost deltas;
  16-core parallel chains.
- Files: `src/hrt_placer/refine_sa.py`.
- Done when: SA further lowers proxy cost on the sample without legality
  violations or timeouts.
- DONE: `refine_sa.py` adds `IncrementalProxy` (a mutable port of the proxy
  cost — verified exact vs `ProxyCost` at init, tracks single-macro moves to
  ~1e-7) and a cold simulated-annealing refiner. A move re-routes only the
  nets on the moved hard macro and re-derives the O(bins) congestion/density
  aggregates, so a chain runs ~1k+ iters/s even on ibm17; 16 per-seed-
  diversified chains run as parallel processes, best legal result wins. Move
  set: Gaussian perturbations + equal/similar-size macro swaps — the compact
  analytical placement leaves ~94% of free nudges overlapping a neighbour, so
  swaps carry the refinement; congestion top-bin heat biases macro selection.
  On the ibm01/03/09/13/17 sample SA lowers proxy 0.5-2.1% (mean -1.1% by
  ProxyCost), every result legal, every benchmark inside its time budget. The
  real evaluator confirms a smaller but positive gain on all five (-0.13% to
  -1.27%): SA optimizes the Pearson-0.98 ProxyCost, so part of the surrogate
  gain does not survive. Gains are modest because the Step-3 analytical
  placement is already near a local optimum; on the largest design (ibm17)
  the real gain is marginal (-0.13%).

### [x] Step 5 — WireMask-EA pass + Klein-4 orientation search — DESCOPED
- Original goal: WireMask-EA HPWL refinement + per-macro orientation search,
  targeting proxy ~<=1.05 over 17 benchmarks.
- DESCOPED after investigating the actual submission interface:
  - Orientation search is infeasible. `place(benchmark)` returns a
    `[num_macros, 2]` positions tensor only (`evaluate.py`); the proxy applies
    *fixed* benchmark pin offsets (`objective.py:_set_placement`) and the DEF
    writer emits the node's default orientation. There is no channel to submit
    orientations, so `orientation.py` would be dead code.
  - WireMask-EA cannot reach <=1.05. The proxy is
    `HPWL + 0.5*density + 0.5*congestion`; HPWL is only ~0.06 of a ~1.30
    proxy (congestion/density dominate). A wirelength-mask-guided method
    cannot move the proxy ~-19% on top of an engine that already beats
    RePlAce on all 17 (AVG 1.3011, -11% mean).
  - With two days to the deadline, the unaddressed Grand-Prize criterion is
    real OpenROAD timing, not further proxy reduction. Effort moves to Step 6.
- WireMask-EA remains a possible Step 7 portfolio member if time allows.

## Day 3 — Grand Prize layer + ship

### [~] Step 6 — Timing weighting + feasibility-gate selection — DESCOPED after /codex review
- `src/hrt_placer/timing.py` + `src/hrt_placer/select.py` exist and are implemented
  but the proxy_budget gate (`select.py:70`, default 0.01) means the timing-weighted
  candidate can only move proxy by ≤1%. Current AVG proxy is 1.2971; PRD target is
  ~1.05 (gap ~25%). Step 6 is structurally incapable of closing the Tier-1 gate.
- Kept in the tree as dead code (low maintenance cost). Not invoked from `placer.py`.
- Future re-enablement only makes sense if a real OpenROAD timer is wired in.

### [~] Step 7 — Hyperparameter-diverse portfolio + congestion-aware engine objective — PARTIAL
- 7a (stderr logging on SA failure), 7b (CUDA determinism pin) and the
  3-candidate portfolio plumbing in placer.py: **DONE**.
- 7d (differentiable congestion + top-k density terms in engine.py loss):
  **NULL RESULT**. Trial on the 5-benchmark sample
  (ibm01/03/09/13/17) on 2026-05-20:
  - baseline (existing tuned engine): AVG proxy 1.1609.
  - proxy_density (top-k density only): AVG 1.2017, +3.51% worse.
  - congestion (top-k density + diff. congestion surrogate): AVG 1.2033,
    +3.65% worse; produced an illegal placement on ibm09 (legalizer could
    not clean up the post-objective layout).
  Mechanism: top-k density makes the spreading gradient sharper, which pushes
  macros apart more aggressively. That raises routing demand on the bins the
  proxy actually measures, so congestion rises faster than density falls.
  The proxy is a balanced metric -- you cannot optimise one term in isolation
  without the others reacting. Proper weight tuning would need an Optuna-style
  sweep we do not have time for inside the 24h deadline.
- Portfolio reverted to baseline-only. ``density_topk_frac`` and
  ``congestion_weight`` knobs stay on ``AnalyticalPlacer`` as future-work
  hyperparameters (default to original behaviour) so a TPE search can revisit
  them properly post-deadline.

### [x] Step 7b — Post-engine soft-macro Adam refiner with per-bench gate
- `src/hrt_placer/refine_soft.py` runs Adam on the soft-macro slice only
  (hard macros frozen) under the engine's smooth HPWL + bin-density-overflow
  loss. ``placer.py`` scores both engine and refined outputs via `ProxyCost`
  per candidate and adopts the refined version only if its proxy is strictly
  lower AND legal -- guaranteed non-regression.
- 5-bench validation (real evaluator, lr=4e-3 n=200 wl=0.2 den=1.0):
  per-bench delta ibm01 -2.57%, ibm03 +1.16% (rejected by gate),
  ibm09 -0.75%, ibm13 -1.34%, ibm17 -2.18%. Post-gate AVG: **1.1456**
  (vs 1.1563 engine-only baseline, **-0.92%**), all 5 legal.

### [~] Step 7c — Weighted-average WL + Nesterov rebuild (engine_v2) — NULL ADD
- `src/hrt_placer/engine_v2.py` (`AnalyticalPlacerV2`) replaces Adam+LSE-HPWL
  with Nesterov-SGD+WA-WL per ePlace/DREAMPlace convention. ePlace's third
  recommendation (density penalty multiplier schedule) skipped for time.
- 5-bench validation 2026-05-21 (lr=0.01, momentum=0.9, n_iters=800):
  ibm01 -0.82%, ibm03 +?, ibm09 +1.7%, ibm13 +0.93%, ibm17 +0.88%.
  AVG +0.51% vs v1 engine alone.
- After the per-bench soft-refine gate, baseline+refine beats v2+refine on
  ALL 5 benches (soft-refine compresses both engines into similar local
  minima, erasing v2's ibm01 edge). Net AVG gain from adding v2 to portfolio:
  **0.00%**. Kept as research artifact for a future port that includes
  ePlace's density-schedule (likely the missing piece) and the
  congestion-aware loss properly TPE-tuned.

### [ ] Step 8 — Tests, doc reconciliation, air-gapped rehearsal, submit
- Tests (`tests/`):
  - `test_legal_all_benchmarks.py` — loop 17 IBMs (+ stub NG45 if available),
    assert `search._is_legal`.
  - `test_determinism.py` — run ibm01 twice with the same seed on CUDA,
    assert byte-identical placements (gated on 7b).
  - `test_timeout_returns_legal.py` — `HRT_TIME_BUDGET=60`, assert legal even when
    the engine is truncated at the deadline.
  - `test_clearance_ng45.py` — synthetic NG45-scale canvas, assert ≥12 µm clearance.
- Doc reconciliation (Codex finding 9 — material rot, not cosmetic):
  - `docs/ARCHITECTURE.md` lists files that DO NOT EXIST: `refine_wiremask.py`,
    `orientation.py`, `portfolio.py`, `external/`, `tests/`, `Dockerfile`. Cut to
    only what shipped.
  - `docs/PRD.md:29-41` still lists WireMask, Klein-4 orientation, GPU soft-macro
    co-opt, portfolio runner, Dockerfile as in-scope. Update In-Scope and
    Out-of-Scope lists to match reality.
  - `README.md:30-31` says no Dockerfile is shipped — keep that contract unless 7e adds one.
- Air-gapped rehearsal: `external/macro-place-challenge-2026/eval_docker/run_eval.sh`
  on 1 IBM + 1 NG45 with `--network none`; verify proxy matches the local run.
- VERIFY the Tier-2 entry cutoff on the official challenge leaderboard / README.
  If cutoff > 1.30 (we miss after 7c+7d), the submission's value is entirely the
  Grand Prize layer (which is unmeasured without OpenROAD). Flag explicitly.
- Done when: tests green; docs consistent; air-gapped run reproduces local proxy;
  submission package zipped and ready for the Google Form.
