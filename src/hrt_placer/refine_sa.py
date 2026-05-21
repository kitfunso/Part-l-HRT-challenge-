"""Hotspot-targeted simulated-annealing refinement.

Takes a legal placement from the analytical engine and lowers the proxy cost
with simulated annealing over hard-macro moves. Every accepted move keeps the
placement legal (zero hard-macro overlap, in-canvas), so the result is legal
by construction.

The cost is maintained incrementally: a single-macro move re-routes only the
nets on the moved macro and re-derives the O(bins) congestion/density
aggregates, instead of re-evaluating all nets. A chain therefore runs
hundreds of thousands of moves inside the time budget. Independent chains run
on separate processes and the best legal result is returned.
"""

import math
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

from .engine import clearance_microns
from .proxy_cost import ProxyCost, _two_pin, _three_pin


class IncrementalProxy:
    """Mutable placement with incremental TILOS proxy-cost bookkeeping.

    Built from a ``Benchmark`` and an initial placement (numpy ``[N, 2]`` of
    micron centers). ``move``/``undo`` apply and revert a single hard-macro
    move while keeping wirelength, density and congestion state consistent.
    """

    def __init__(self, benchmark, placement):
        pc = ProxyCost(benchmark)
        self.pc = pc
        # macro-to-macro clearance (PRD): a candidate box is inflated by the
        # full clearance so every other (real-size) hard macro stays clear.
        clr_um = clearance_microns(pc.W, pc.H)
        self.clr_x = clr_um / pc.W
        self.clr_y = clr_um / pc.H
        self.rows, self.cols = pc.rows, pc.cols
        self.gw, self.gh = pc.gw, pc.gh
        self.num_hard = pc.num_hard
        self.num_macros = pc.num_macros
        self.num_nets = pc.num_nets

        self.pos = np.asarray(placement, dtype=np.float64).copy()
        self.sizes = pc.sizes.numpy().astype(np.float64)
        self.half = self.sizes / 2.0
        self.fixed = pc.macro_fixed.numpy().astype(bool)

        # per-pin static data
        self.owner = pc.owner
        self.offx, self.offy = pc.offx, pc.offy
        self.net_ptr = pc.net_ptr
        self.net_w = pc.net_w
        self.ports = pc.ports

        # per-pin live coordinates / bin indices
        self._init_pins()

        # macro -> its pins / nets
        net_of_pin = np.repeat(np.arange(self.num_nets),
                               np.diff(self.net_ptr))
        macro_pins = [[] for _ in range(self.num_hard)]
        macro_nets = [set() for _ in range(self.num_hard)]
        for k, o in enumerate(self.owner):
            if o < self.num_hard:
                macro_pins[o].append(k)
                macro_nets[o].add(int(net_of_pin[k]))
        self.macro_pins = [np.array(p, dtype=np.int64) for p in macro_pins]
        self.macro_nets = [np.array(sorted(s), dtype=np.int64)
                           for s in macro_nets]

        # smoothing windows / box-filter index tables
        r = pc.smooth_range
        self.r = r
        ci = np.arange(self.cols)
        ri = np.arange(self.rows)
        self.win_v = (np.minimum(self.cols - 1, ci + r)
                      - np.maximum(0, ci - r) + 1).astype(np.float64)
        self.win_h = (np.minimum(self.rows - 1, ri + r)
                      - np.maximum(0, ri - r) + 1).astype(np.float64)
        self.bx_hi = np.minimum(self.cols, ci + r + 1)
        self.bx_lo = np.maximum(0, ci - r)
        self.br_hi = np.minimum(self.rows, ri + r + 1)
        self.br_lo = np.maximum(0, ri - r)

        self._rebuild()

    # ---- (re)build full state ----------------------------------------

    def _init_pins(self):
        all_pos = np.concatenate([self.pos, self.ports], axis=0)
        self.p_x = all_pos[self.owner, 0] + self.offx
        self.p_y = all_pos[self.owner, 1] + self.offy
        self.p_col = np.clip((self.p_x / self.gw).astype(np.int64),
                             0, self.cols - 1)
        self.p_row = np.clip((self.p_y / self.gh).astype(np.int64),
                             0, self.rows - 1)

    def _rebuild(self):
        """Recompute all grids and per-net state from current positions."""
        self.Hr = np.zeros((self.rows, self.cols))
        self.Vr = np.zeros((self.rows, self.cols))
        for i in range(self.num_nets):
            self._route_net(i, 1.0)
        self.Hm = np.zeros((self.rows, self.cols))
        self.Vm = np.zeros((self.rows, self.cols))
        for m in range(self.num_hard):
            self._route_macro(m, 1.0)
        self.occ = self._build_occ()
        self.net_span = np.array([self._net_span(i)
                                  for i in range(self.num_nets)])
        self.wl_sum = float((self.net_w * self.net_span).sum())

    def _build_occ(self):
        ci = np.arange(self.cols)
        ri = np.arange(self.rows)
        bx_lo, bx_hi = ci * self.gw, (ci + 1) * self.gw
        by_lo, by_hi = ri * self.gh, (ri + 1) * self.gh
        x_lo = (self.pos[:, 0] - self.half[:, 0])[:, None]
        x_hi = (self.pos[:, 0] + self.half[:, 0])[:, None]
        y_lo = (self.pos[:, 1] - self.half[:, 1])[:, None]
        y_hi = (self.pos[:, 1] + self.half[:, 1])[:, None]
        ox = np.clip(np.minimum(x_hi, bx_hi) - np.maximum(x_lo, bx_lo), 0, None)
        oy = np.clip(np.minimum(y_hi, by_hi) - np.maximum(y_lo, by_lo), 0, None)
        return np.einsum("nr,nc->rc", oy, ox)

    # ---- per-element routing / occupancy -----------------------------

    def _route_net(self, i, sign):
        s, e = self.net_ptr[i], self.net_ptr[i + 1]
        if e - s < 2:
            return
        w = sign * (float(self.net_w[i]) if i < len(self.net_w) else 1.0)
        rr = self.p_row[s:e]
        cc = self.p_col[s:e]
        cells = list({(int(r), int(c)) for r, c in zip(rr, cc)})
        source = (int(rr[0]), int(cc[0]))
        n = len(cells)
        if n == 2:
            sink = cells[0] if cells[1] == source else cells[1]
            _two_pin(self.Hr, self.Vr, source, sink, w)
        elif n == 3:
            _three_pin(self.Hr, self.Vr, cells, w)
        elif n > 3:
            for c in cells:
                if c != source:
                    _two_pin(self.Hr, self.Vr, source, c, w)

    def _route_macro(self, m, sign):
        mx, my = self.pos[m, 0], self.pos[m, 1]
        mw, mh = float(self.sizes[m, 0]), float(self.sizes[m, 1])
        gw, gh = self.gw, self.gh
        x_min, x_max = mx - mw / 2, mx + mw / 2
        y_min, y_max = my - mh / 2, my + mh / 2
        bl_col = int(np.clip(math.floor(x_min / gw), 0, self.cols - 1))
        ur_col = int(np.clip(math.floor(x_max / gw), 0, self.cols - 1))
        bl_row = int(np.clip(math.floor(y_min / gh), 0, self.rows - 1))
        ur_row = int(np.clip(math.floor(y_max / gh), 0, self.rows - 1))
        va = sign * self.pc.v_alloc
        ha = sign * self.pc.h_alloc
        part_v = part_h = False
        for r in range(bl_row, ur_row + 1):
            for c in range(bl_col, ur_col + 1):
                cx0, cx1 = c * gw, (c + 1) * gw
                cy0, cy1 = r * gh, (r + 1) * gh
                xd = min(x_max, cx1) - max(x_min, cx0)
                yd = min(y_max, cy1) - max(y_min, cy0)
                if not (xd > 0 and yd > 0):
                    xd = yd = 0.0
                if ur_row != bl_row and r in (bl_row, ur_row) \
                        and abs(yd - gh) > 1e-5:
                    part_v = True
                if ur_col != bl_col and c in (bl_col, ur_col) \
                        and abs(xd - gw) > 1e-5:
                    part_h = True
                self.Vm[r, c] += xd * va
                self.Hm[r, c] += yd * ha
        if part_v:
            for c in range(bl_col, ur_col + 1):
                cx0, cx1 = c * gw, (c + 1) * gw
                cy0, cy1 = ur_row * gh, (ur_row + 1) * gh
                xd = min(x_max, cx1) - max(x_min, cx0)
                yd = min(y_max, cy1) - max(y_min, cy0)
                if xd > 0 and yd > 0:
                    self.Vm[ur_row, c] -= xd * va
        if part_h:
            for r in range(bl_row, ur_row + 1):
                cx0, cx1 = ur_col * gw, (ur_col + 1) * gw
                cy0, cy1 = r * gh, (r + 1) * gh
                xd = min(x_max, cx1) - max(x_min, cx0)
                yd = min(y_max, cy1) - max(y_min, cy0)
                if xd > 0 and yd > 0:
                    self.Hm[r, ur_col] -= yd * ha

    def _occ_macro(self, m, sign):
        ci = np.arange(self.cols)
        ri = np.arange(self.rows)
        bx_lo, bx_hi = ci * self.gw, (ci + 1) * self.gw
        by_lo, by_hi = ri * self.gh, (ri + 1) * self.gh
        hx, hy = self.half[m, 0], self.half[m, 1]
        mx, my = self.pos[m, 0], self.pos[m, 1]
        ox = np.clip(np.minimum(mx + hx, bx_hi) - np.maximum(mx - hx, bx_lo),
                     0, None)
        oy = np.clip(np.minimum(my + hy, by_hi) - np.maximum(my - hy, by_lo),
                     0, None)
        self.occ += sign * np.outer(oy, ox)

    def _net_span(self, i):
        s, e = self.net_ptr[i], self.net_ptr[i + 1]
        if e - s < 1:
            return 0.0
        xs, ys = self.p_x[s:e], self.p_y[s:e]
        return (xs.max() - xs.min()) + (ys.max() - ys.min())

    def _update_macro_pins(self, m):
        pins = self.macro_pins[m]
        if pins.size == 0:
            return
        self.p_x[pins] = self.pos[m, 0] + self.offx[pins]
        self.p_y[pins] = self.pos[m, 1] + self.offy[pins]
        self.p_col[pins] = np.clip((self.p_x[pins] / self.gw).astype(np.int64),
                                   0, self.cols - 1)
        self.p_row[pins] = np.clip((self.p_y[pins] / self.gh).astype(np.int64),
                                   0, self.rows - 1)

    # ---- cost aggregates ---------------------------------------------

    def _smooth_v(self, g):
        if self.r <= 0:
            return g
        g2 = g / self.win_v
        cs = np.concatenate([np.zeros((g.shape[0], 1)),
                             np.cumsum(g2, axis=1)], axis=1)
        return cs[:, self.bx_hi] - cs[:, self.bx_lo]

    def _smooth_h(self, g):
        if self.r <= 0:
            return g
        g2 = g / self.win_h[:, None]
        cs = np.concatenate([np.zeros((1, g.shape[1])),
                             np.cumsum(g2, axis=0)], axis=0)
        return cs[self.br_hi, :] - cs[self.br_lo, :]

    def congestion(self, return_grids=False):
        H = self._smooth_h(self.Hr / self.pc.grid_h_routes)
        V = self._smooth_v(self.Vr / self.pc.grid_v_routes)
        H = H + self.Hm / self.pc.grid_h_routes
        V = V + self.Vm / self.pc.grid_v_routes
        allc = np.concatenate([V.ravel(), H.ravel()])
        cnt = int(math.floor(len(allc) * 0.05))
        if cnt == 0:
            cong = float(allc.max())
        else:
            cong = float(np.partition(allc, -cnt)[-cnt:].sum() / cnt)
        if return_grids:
            return cong, H, V
        return cong

    def density(self):
        density = (self.occ / self.pc.bin_area).ravel()
        nz = density[density != 0.0]
        if nz.size == 0:
            return 0.0
        if self.pc.num_bins < 10:
            return 0.5 * float(nz.mean())
        k = max(1, int(math.floor(0.1 * self.pc.num_bins)))
        kk = min(k, nz.size)
        top = np.partition(nz, -kk)[-kk:]
        return 0.5 * float(top.sum()) / k

    def wirelength(self):
        return (self.pc.cal_wl * self.wl_sum
                / ((self.pc.W + self.pc.H) * max(1, self.num_nets)))

    def cost(self):
        return (self.wirelength() + 0.5 * self.density()
                + 0.5 * self.pc.cal_cong * self.congestion())

    # ---- moves --------------------------------------------------------

    def box_free(self, x, y, hx, hy, exclude):
        """True if box (center x,y; half hx,hy) is in-canvas and at least
        12 um clear of every hard macro except those in ``exclude``."""
        if (x - hx < -1e-9 or x + hx > self.pc.W + 1e-9
                or y - hy < -1e-9 or y + hy > self.pc.H + 1e-9):
            return False
        nh = self.num_hard
        ix, iy = hx + self.clr_x, hy + self.clr_y
        px, py = self.pos[:nh, 0], self.pos[:nh, 1]
        hwx, hwy = self.half[:nh, 0], self.half[:nh, 1]
        sep = ((x + ix <= px - hwx) | (x - ix >= px + hwx)
               | (y + iy <= py - hwy) | (y - iy >= py + hwy))
        for m in exclude:
            sep[m] = True
        return bool(sep.all())

    def legal(self, m, nx, ny):
        """True if macro ``m`` at ``(nx, ny)`` is in-canvas and overlap-free."""
        return self.box_free(nx, ny, self.half[m, 0], self.half[m, 1], (m,))

    def move(self, m, nx, ny):
        """Apply a move of macro ``m`` to ``(nx, ny)``; return (cost, token)."""
        affected = self.macro_nets[m]
        token = (m, self.pos[m].copy(),
                 {int(i): self.net_span[i] for i in affected})
        for i in affected:
            self._route_net(i, -1.0)
        self._route_macro(m, -1.0)
        self._occ_macro(m, -1.0)
        self.pos[m, 0], self.pos[m, 1] = nx, ny
        self._update_macro_pins(m)
        for i in affected:
            self._route_net(i, 1.0)
        self._route_macro(m, 1.0)
        self._occ_macro(m, 1.0)
        for i in affected:
            new = self._net_span(i)
            self.wl_sum += self.net_w[i] * (new - self.net_span[i])
            self.net_span[i] = new
        return self.cost(), token

    def undo(self, token):
        m, old_pos, old_spans = token
        affected = self.macro_nets[m]
        for i in affected:
            self._route_net(i, -1.0)
        self._route_macro(m, -1.0)
        self._occ_macro(m, -1.0)
        self.pos[m] = old_pos
        self._update_macro_pins(m)
        for i in affected:
            self._route_net(i, 1.0)
        self._route_macro(m, 1.0)
        self._occ_macro(m, 1.0)
        for i in affected:
            self.wl_sum += self.net_w[i] * (old_spans[int(i)]
                                            - self.net_span[i])
            self.net_span[i] = old_spans[int(i)]


def _run_chain(benchmark, placement, budget, seed):
    """One SA chain. Returns (best_cost, best_placement_numpy).

    Move set is perturbations (Gaussian nudge) plus equal/similar-size macro
    swaps. From the compact analytical placement almost every free nudge
    overlaps a neighbour; swaps of like-sized macros stay legal and carry the
    refinement.
    """
    ip = IncrementalProxy(benchmark, placement)
    rng = np.random.default_rng(seed)
    movable = np.array([m for m in range(ip.num_hard) if not ip.fixed[m]],
                       dtype=np.int64)
    cur = ip.cost()
    best_cost = cur
    best_pos = ip.pos.copy()
    if movable.size < 2:
        return best_cost, best_pos

    # per-chain strategy variation so 16 chains explore differently
    cfg = np.random.default_rng(seed + 99991)
    swap_frac = float(cfg.uniform(0.30, 0.65))
    sig_hi = float(cfg.uniform(2.2, 3.4))
    sig_lo = float(cfg.uniform(1.0, 1.8))
    hot_bias = float(cfg.uniform(0.6, 0.85))

    bin_d = max(ip.gw, ip.gh)
    areas = ip.sizes[:, 0] * ip.sizes[:, 1]
    size_sorted = movable[np.argsort(areas[movable], kind="stable")]
    nsz = size_sorted.size
    rank = np.empty(ip.num_hard, dtype=np.int64)
    rank[size_sorted] = np.arange(nsz)
    window = 12

    def partner(a):
        j = int(rank[a]) + int(rng.integers(-window, window + 1))
        b = int(size_sorted[min(max(j, 0), nsz - 1)])
        return b if b != a else int(size_sorted[(rank[a] + 1) % nsz])

    def try_swap(a, b):
        ax, ay = ip.pos[a, 0], ip.pos[a, 1]
        bx, by = ip.pos[b, 0], ip.pos[b, 1]
        hax, hay = ip.half[a, 0], ip.half[a, 1]
        hbx, hby = ip.half[b, 0], ip.half[b, 1]
        if not ip.box_free(bx, by, hax, hay, (a, b)):
            return None
        if not ip.box_free(ax, ay, hbx, hby, (a, b)):
            return None
        if not (bx + hax <= ax - hbx or bx - hax >= ax + hbx
                or by + hay <= ay - hby or by - hay >= ay + hby):
            return None
        _, t1 = ip.move(a, bx, by)
        c2, t2 = ip.move(b, ax, ay)
        return c2, (t1, t2)

    # cold schedule: this is refinement of an already-good placement, so the
    # walk must stay near the start (near-greedy, only tiny uphill accepted).
    t0 = cur * float(10 ** cfg.uniform(-4.5, -3.7))
    t_end = t0 * 0.02

    # --- basin hopping -------------------------------------------------
    # Vanilla SA cools once over the whole budget and does small moves, so
    # it polishes a single basin and stalls. Basin hopping uses the full
    # time budget: when the search has not improved ``best_cost`` for
    # ``stuck_limit`` iterations it (a) optionally re-centres on the global
    # best, (b) applies a forced legal swap cascade -- a big jump to a
    # different basin -- and (c) reheats the temperature. ``best_pos`` is
    # preserved across every hop, so a hop into a worse basin never costs
    # us the best placement found. The per-cycle cooling lets each basin
    # be properly annealed before the next hop.
    cycle_len = max(20000, movable.size * 60)
    stuck_limit = max(8000, movable.size * 20)
    kick_size = max(5, movable.size // 6)

    start = time.time()
    deadline = start + budget
    hot = movable
    iters = 0
    stuck = 0
    since_kick = 0
    n_kicks = 0

    while True:
        iters += 1
        since_kick += 1
        if iters % 128 == 0 and time.time() > deadline:
            break
        # per-cycle cooling: temperature resets hot after each basin hop.
        cyc = min(1.0, since_kick / cycle_len)
        temp = t0 * (t_end / t0) ** cyc

        if iters % 256 == 1:
            _, Hg, Vg = ip.congestion(return_grids=True)
            heat_grid = np.maximum(Hg, Vg)
            cols = np.clip((ip.pos[movable, 0] / ip.gw).astype(np.int64),
                           0, ip.cols - 1)
            rows = np.clip((ip.pos[movable, 1] / ip.gh).astype(np.int64),
                           0, ip.rows - 1)
            order = np.argsort(-heat_grid[rows, cols])
            hot = movable[order[:max(1, movable.size // 3)]]

        a = (int(hot[rng.integers(hot.size)]) if rng.random() < hot_bias
             else int(movable[rng.integers(movable.size)]))

        if rng.random() > swap_frac:
            sigma = bin_d * (sig_hi - (sig_hi - sig_lo) * cyc)
            nx = ip.pos[a, 0] + rng.normal(0, sigma)
            ny = ip.pos[a, 1] + rng.normal(0, sigma)
            if not ip.legal(a, nx, ny):
                continue
            c, tok = ip.move(a, nx, ny)
            undo = (tok,)
        else:
            res = try_swap(a, partner(a))
            if res is None:
                continue
            c, undo = res

        delta = c - cur
        if delta <= 0 or rng.random() < math.exp(-delta / temp):
            cur = c
            if cur < best_cost - 1e-12:
                best_cost = cur
                best_pos = ip.pos.copy()
                stuck = 0
            else:
                stuck += 1
        else:
            for tok in reversed(undo):
                ip.undo(tok)
            stuck += 1

        # Basin hop when the search has stalled.
        if stuck >= stuck_limit:
            n_kicks += 1
            # Every 4th hop, re-centre on the global best so a run of bad
            # basins does not drift the search away permanently.
            if n_kicks % 4 == 0:
                ip = IncrementalProxy(benchmark, best_pos)
                cur = ip.cost()
            # Forced legal swap cascade: a big jump to a new basin. Swaps
            # of like-sized macros stay legal; force-accept (no undo).
            for _ in range(kick_size):
                a = int(movable[rng.integers(movable.size)])
                res = try_swap(a, partner(a))
                if res is not None:
                    cur = res[0]
            stuck = 0
            since_kick = 0

    return best_cost, best_pos


def refine(benchmark, placement, time_budget=30.0, n_chains=None, seed=0,
           verbose=False):
    """Refine ``placement`` with parallel SA chains.

    ``placement`` is a ``[num_macros, 2]`` micron-coordinate tensor (the
    analytical engine's output). Returns an improved legal placement tensor;
    never returns a placement worse (by proxy cost) than the input.
    """
    pl = placement.detach().cpu().numpy().astype(np.float64)
    if n_chains is None:
        n_chains = min(16, os.cpu_count() or 1)

    results = []
    try:
        with ProcessPoolExecutor(max_workers=n_chains) as ex:
            futs = [ex.submit(_run_chain, benchmark, pl, time_budget, seed + i)
                    for i in range(n_chains)]
            for f in futs:
                results.append(f.result())
    except Exception as exc:  # fall back to a single in-process chain
        if verbose:
            print(f"  [refine] parallel failed ({exc}); single chain")
        results = [_run_chain(benchmark, pl, time_budget, seed)]

    # chains report their own (faithful) proxy cost, so ranking needs no
    # re-evaluation; one ProxyCost call scores the untouched input.
    base = ProxyCost(benchmark)(placement)["proxy_cost"]
    best_pos, best = pl, base
    for cost, pos in results:
        if cost < best - 1e-9:
            best, best_pos = cost, pos
    if verbose:
        print(f"  [refine] {n_chains} chains: {base:.4f} -> {best:.4f} "
              f"({(best / base - 1) * 100:+.2f}%)")
    return torch.from_numpy(np.asarray(best_pos)).float()


def main():
    """CLI: engine then SA refinement, before/after proxy on a sample."""
    import sys

    repo = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    challenge = os.path.join(repo, "external", "macro-place-challenge-2026")
    sys.path.insert(0, os.path.join(repo, "src"))
    os.chdir(challenge)
    from macro_place.loader import load_benchmark_from_dir
    from macro_place.utils import validate_placement

    from hrt_placer.engine import AnalyticalPlacer
    from hrt_placer.proxy_cost import ProxyCost

    base = "external/MacroPlacement/Testcases/ICCAD04"
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    budget = 30.0
    use_real = "--real" in sys.argv
    for a in sys.argv[1:]:
        if a.startswith("--budget="):
            budget = float(a.split("=")[1])
    names = args or ["ibm01", "ibm03", "ibm09", "ibm13", "ibm17"]

    compute_proxy_cost = None
    if use_real:
        from macro_place.objective import compute_proxy_cost

    print(f"scoring: {'real evaluator' if use_real else 'ProxyCost (faithful)'}")
    print(f"{'bench':>7} {'engine':>9} {'+SA':>9} {'gain':>8} "
          f"{'valid':>6} {'t(s)':>7}")
    gains = []
    for name in names:
        bm, plc = load_benchmark_from_dir(f"{base}/{name}")
        placement = AnalyticalPlacer().place(bm)
        pc = ProxyCost(bm)

        def score(p):
            if use_real:
                return compute_proxy_cost(p, bm, plc)["proxy_cost"]
            return pc(p)["proxy_cost"]

        c0 = score(placement)
        t0 = time.time()
        refined = refine(bm, placement, time_budget=budget)
        dt = time.time() - t0
        c1 = score(refined)
        valid, viol = validate_placement(refined, bm)
        gains.append(c1 / c0 - 1.0)
        print(f"{name:>7} {c0:9.4f} {c1:9.4f} "
              f"{(c1 / c0 - 1) * 100:+7.2f}% {str(valid):>6} {dt:7.1f}")
        if not valid:
            print(f"        violations: {viol}")
    if gains:
        print(f"{'AVG':>7} {'':9} {'':9} {np.mean(gains) * 100:+7.2f}%")


if __name__ == "__main__":
    main()
