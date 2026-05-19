"""Bayesian (TPE) hyperparameter search over the analytical placer.

Wraps ``AnalyticalPlacer`` as a black-box whose hyperparameters are tuned to
minimize the (faithful) TILOS proxy cost. The objective for each trial is the
mean of ``proxy(trial) / proxy(stock)`` over a sample of benchmarks, so every
benchmark contributes equally regardless of absolute cost; a value below 1.0
means the tuned config beats stock defaults.

The optimizer is a self-contained Tree-structured Parzen Estimator: no external
dependency, so it runs both at dev time (to pick defaults) and at submission
runtime (to self-tune per design, offline, within the hour budget).
"""

import math

import numpy as np
import torch

from .engine import AnalyticalPlacer
from .proxy_cost import ProxyCost

# name -> (low, high, kind); kind in {"lin", "log", "int"}
DEFAULT_SPACE = {
    "lr": (0.002, 0.04, "log"),
    "gamma_start": (0.015, 0.14, "lin"),
    "gamma_end": (0.002, 0.025, "lin"),
    "density_weight": (0.5, 28.0, "log"),
    "overlap_weight": (2.0, 40.0, "log"),
    "n_iters": (150, 900, "int"),
}


def _from_unit(u, lo, hi, kind):
    u = min(1.0, max(0.0, float(u)))
    if kind == "log":
        v = math.exp(math.log(lo) + u * (math.log(hi) - math.log(lo)))
    else:
        v = lo + u * (hi - lo)
    return int(round(v)) if kind == "int" else v


def _is_legal(placement, benchmark, ov_tol=1e-3, edge_tol=1e-3):
    """Zero hard-macro overlap and every macro inside the canvas."""
    pos = placement.float()
    half = benchmark.macro_sizes.float() / 2.0
    cw, ch = float(benchmark.canvas_width), float(benchmark.canvas_height)
    lo, hi = pos - half, pos + half
    if (lo[:, 0] < -edge_tol).any() or (lo[:, 1] < -edge_tol).any():
        return False
    if (hi[:, 0] > cw + edge_tol).any() or (hi[:, 1] > ch + edge_tol).any():
        return False
    nh = benchmark.num_hard_macros
    if nh > 1:
        p, h = pos[:nh], half[:nh]
        xl, xr = p[:, 0] - h[:, 0], p[:, 0] + h[:, 0]
        yl, yr = p[:, 1] - h[:, 1], p[:, 1] + h[:, 1]
        ox = (torch.minimum(xr[:, None], xr[None, :])
              - torch.maximum(xl[:, None], xl[None, :]))
        oy = (torch.minimum(yr[:, None], yr[None, :])
              - torch.maximum(yl[:, None], yl[None, :]))
        ov = (ox > ov_tol) & (oy > ov_tol)
        ov.fill_diagonal_(False)
        if ov.any():
            return False
    return True


class HyperparamSearch:
    """TPE search over ``AnalyticalPlacer`` hyperparameters.

    ``benchmarks`` is a list of loaded ``Benchmark`` objects (the search
    sample). ``run`` returns the best config and its objective value.
    """

    def __init__(self, benchmarks, space=None, seed=0,
                 gamma=0.25, n_candidates=48, bandwidth=0.22,
                 illegal_penalty=2.0):
        self.benchmarks = list(benchmarks)
        self.space = space or DEFAULT_SPACE
        self.names = list(self.space.keys())
        self.dim = len(self.space)
        self.rng = np.random.default_rng(seed)
        self.gamma = gamma
        self.n_candidates = n_candidates
        self.bw = bandwidth
        self.illegal_penalty = illegal_penalty
        self.trials = []        # list of (uvec, score, config, legal)
        self._pc = None
        self._baseline = None

    # ---- config encoding ---------------------------------------------

    def _config(self, uvec):
        cfg = {}
        for i, name in enumerate(self.names):
            cfg[name] = _from_unit(uvec[i], *self.space[name])
        # the engine anneals gamma_start -> gamma_end; keep it decreasing
        if "gamma_end" in cfg and "gamma_start" in cfg:
            cfg["gamma_end"] = min(cfg["gamma_end"], 0.9 * cfg["gamma_start"])
        return cfg

    # ---- objective ----------------------------------------------------

    def _ensure_baseline(self, verbose=False):
        if self._baseline is not None:
            return
        self._pc = [ProxyCost(bm) for bm in self.benchmarks]
        self._baseline = []
        for i, bm in enumerate(self.benchmarks):
            placer = AnalyticalPlacer()  # stock defaults
            placement = placer.place(bm)
            cost = self._pc[i](placement)["proxy_cost"]
            self._baseline.append(cost)
            if verbose:
                print(f"  baseline {bm.name:>7}: {cost:.4f}")

    def evaluate(self, config, seed=0):
        """Mean proxy/baseline ratio for ``config`` over the sample."""
        self._ensure_baseline()
        ratios, legal_all = [], True
        for i, bm in enumerate(self.benchmarks):
            placer = AnalyticalPlacer(seed=seed, **config)
            placement = placer.place(bm)
            if not _is_legal(placement, bm):
                legal_all = False
                ratios.append(self.illegal_penalty)
                continue
            cost = self._pc[i](placement)["proxy_cost"]
            ratios.append(cost / max(1e-9, self._baseline[i]))
        return float(np.mean(ratios)), legal_all

    # ---- TPE ----------------------------------------------------------

    def _kde(self, X, sample):
        """Product-Gaussian kernel density of ``X`` under ``sample`` points."""
        if len(sample) == 0:
            return np.ones(len(X))
        h = self.bw
        diff = (X[:, None, :] - sample[None, :, :]) / h
        kern = np.exp(-0.5 * diff ** 2) / (h * math.sqrt(2.0 * math.pi))
        return kern.prod(axis=2).mean(axis=1)

    def _suggest(self):
        U = np.stack([t[0] for t in self.trials])
        scores = np.array([t[1] for t in self.trials])
        order = np.argsort(scores)
        n_good = max(1, int(math.ceil(self.gamma * len(scores))))
        good = U[order[:n_good]]
        bad = U[order[n_good:]]

        cands = []
        for _ in range(self.n_candidates):
            base = good[self.rng.integers(len(good))]
            cands.append(np.clip(base + self.rng.normal(0, self.bw, self.dim),
                                 0.0, 1.0))
        for _ in range(max(1, self.n_candidates // 6)):
            cands.append(self.rng.random(self.dim))
        cands = np.array(cands)

        ei = self._kde(cands, good) / (self._kde(cands, bad) + 1e-12)
        return cands[int(np.argmax(ei))]

    def run(self, n_trials=30, n_startup=8, verbose=True):
        """Run the search; return (best_config, best_score)."""
        self._ensure_baseline(verbose=verbose)
        for t in range(n_trials):
            if len(self.trials) < n_startup:
                uvec = self.rng.random(self.dim)
            else:
                uvec = self._suggest()
            config = self._config(uvec)
            score, legal = self.evaluate(config)
            self.trials.append((uvec, score, config, legal))
            if verbose:
                best = min(s for _, s, _, _ in self.trials)
                tag = "" if legal else " (illegal)"
                print(f"  trial {t + 1:>3}/{n_trials}: "
                      f"score={score:.4f}  best={best:.4f}{tag}")
        best = min(self.trials, key=lambda x: x[1])
        return best[2], best[1]

    def best(self):
        b = min(self.trials, key=lambda x: x[1])
        return b[2], b[1]


def main():
    """CLI: search on an IBM sample and report stock vs tuned proxy."""
    import os
    import sys

    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    challenge = os.path.join(repo, "external", "macro-place-challenge-2026")
    sys.path.insert(0, os.path.join(repo, "src"))
    os.chdir(challenge)
    from macro_place.loader import load_benchmark_from_dir

    base = "external/MacroPlacement/Testcases/ICCAD04"
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    n_trials = 30
    for a in sys.argv[1:]:
        if a.startswith("--trials="):
            n_trials = int(a.split("=")[1])
    names = args or ["ibm01", "ibm03", "ibm04", "ibm07", "ibm09",
                     "ibm11", "ibm12", "ibm14", "ibm16", "ibm17"]

    print(f"Loading {len(names)} benchmarks: {', '.join(names)}")
    benchmarks = [load_benchmark_from_dir(f"{base}/{n}")[0] for n in names]

    search = HyperparamSearch(benchmarks, seed=0)
    print("Stock baselines:")
    best_config, best_score = search.run(n_trials=n_trials)

    print(f"\nBest objective (mean proxy/stock): {best_score:.4f}  "
          f"({(best_score - 1.0) * 100:+.1f}% vs stock)")
    print("Best config:")
    for k, v in best_config.items():
        print(f"  {k:>16}: {v}")

    print("\nPer-benchmark stock -> tuned proxy:")
    for i, bm in enumerate(benchmarks):
        placer = AnalyticalPlacer(seed=0, **best_config)
        tuned = search._pc[i](placer.place(bm))["proxy_cost"]
        stock = search._baseline[i]
        print(f"  {bm.name:>7}: {stock:.4f} -> {tuned:.4f}  "
              f"({(tuned / stock - 1) * 100:+.1f}%)")


if __name__ == "__main__":
    main()
