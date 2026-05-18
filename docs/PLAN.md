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

### [ ] Step 3 — Exact proxy cost as the search objective
- Goal: wire the exact TILOS proxy cost as a black-box objective; add the
  MOTPE/Bayesian search loop over DREAMPlace hyperparameters.
- Files: `src/hrt_placer/proxy_cost.py`, `src/hrt_placer/search.py`.
- Done when: BO search measurably lowers proxy cost vs stock DREAMPlace on a
  10-benchmark sample.

### [ ] Step 4 — Hotspot-targeted SA refinement
- Goal: SA that attacks the top-5%/top-10% bins, with incremental cost deltas;
  16-core parallel chains.
- Files: `src/hrt_placer/refine_sa.py`.
- Done when: SA further lowers proxy cost on the sample without legality
  violations or timeouts.

### [ ] Step 5 — WireMask-EA pass + Klein-4 orientation search
- Goal: add the WireMask-EA HPWL refinement and per-macro orientation search.
- Files: `src/hrt_placer/refine_wiremask.py`, `src/hrt_placer/orientation.py`,
  `external/WireMask-BBO/`.
- Done when: combined pipeline reaches proxy ~<=1.05 averaged over 17 benchmarks.

## Day 3 — Grand Prize layer + ship

### [ ] Step 6 — Timing weighting + feasibility-gate selection
- Goal: topology-derived net criticality weighting; candidate selection that
  favors timing-safe placements over marginally-lower proxy cost.
- Files: `src/hrt_placer/timing.py`, `src/hrt_placer/select.py`.
- Done when: selected placements pass an OpenROAD/Hier-RTLMP timing sanity
  check vs the SA/RePlAce baselines.

### [ ] Step 7 — Portfolio runner + full validation
- Goal: budget allocation across stages with a hard per-benchmark timeout;
  full 17-benchmark run.
- Files: `src/hrt_placer/portfolio.py`, `tests/`.
- Done when: all 17 benchmarks complete legally within the 1-hour cap.

### [ ] Step 8 — Reproducible packaging + submission
- Goal: finalize Dockerfile, pin submodules, vendor licenses, write README.
- Files: `Dockerfile`, `requirements.txt`, `README.md`.
- Done when: a clean Docker build reproduces the validation results offline,
  and the submission package is ready for the Google Form.
