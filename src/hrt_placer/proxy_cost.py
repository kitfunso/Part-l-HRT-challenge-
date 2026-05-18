"""From-Benchmark reimplementation of the TILOS proxy cost.

The challenge evaluator (``macro_place.objective.compute_proxy_cost``) needs the
``PlacementCost`` object, which ``place()`` never receives. To score and refine
candidates at runtime — and on unseen Tier 2 designs — the placer needs its own
proxy-cost model built purely from the ``Benchmark`` tensors.

This is a faithful port of ``plc_client_os.py``:
  * wirelength  — per-net HPWL over pin endpoints (get_wirelength / get_cost)
  * density     — exact rect/bin overlap, 0.5 * top-10% of non-zero bins
  * congestion  — grid routing of every net (2-pin L-routes, 3-pin shapes,
                  >3-pin star split), macro routing demand, the spreading
                  smoothing, then ABU-5 over the combined H/V maps

Wirelength keeps one calibration scale: the real ``get_cost`` divides by
``plc.net_cnt`` which is not exposed on the Benchmark, so our normalization is
off by a benchmark-specific constant (~1.19x) that ``cal_wl`` absorbs.
"""

import math

import numpy as np
import torch

# Macro routing allocation and smoothing are identical across all 17 IBM
# .plc files and are not carried on the Benchmark. Defaults below reproduce
# them; for other tech the ratios vs routes_per_micron are a fallback.
_IBM_SMOOTH_RANGE = 2
_H_ALLOC_RATIO = 30.304 / 65.957
_V_ALLOC_RATIO = 71.304 / 106.957


def _net_pin_lists(benchmark):
    """Per-net flat pin arrays (owner, x-offset, y-offset), driver pin first."""
    num_hard = benchmark.num_hard_macros
    use_pin = (
        len(benchmark.net_pin_nodes) == benchmark.num_nets
        and benchmark.num_nets > 0
    )
    owner, offx, offy, ptr = [], [], [], [0]
    src = benchmark.net_pin_nodes if use_pin else benchmark.net_nodes
    for net in src:
        if net.numel() == 0:
            ptr.append(len(owner))
            continue
        if use_pin:
            for o, pidx in net.tolist():
                owner.append(o)
                if o < num_hard and o < len(benchmark.macro_pin_offsets):
                    off = benchmark.macro_pin_offsets[o]
                    if 0 <= pidx < off.shape[0]:
                        offx.append(float(off[pidx, 0]))
                        offy.append(float(off[pidx, 1]))
                    else:
                        offx.append(0.0)
                        offy.append(0.0)
                else:
                    offx.append(0.0)
                    offy.append(0.0)
        else:
            for o in net.tolist():
                owner.append(o)
                offx.append(0.0)
                offy.append(0.0)
        ptr.append(len(owner))
    return (np.array(owner, dtype=np.int64),
            np.array(offx, dtype=np.float64),
            np.array(offy, dtype=np.float64),
            np.array(ptr, dtype=np.int64))


class ProxyCost:
    """Proxy-cost model reconstructed from a ``Benchmark``.

    ``cost = wirelength + 0.5 * density + 0.5 * congestion`` (the 0.5 on
    density is already inside ``density_cost``, matching the real evaluator).
    """

    def __init__(self, benchmark, smooth_range=None,
                 hrouting_alloc=None, vrouting_alloc=None,
                 cal_wl=0.84, cal_cong=1.11):
        self.W = float(benchmark.canvas_width)
        self.H = float(benchmark.canvas_height)
        self.num_macros = benchmark.num_macros
        self.num_hard = benchmark.num_hard_macros
        self.num_nets = benchmark.num_nets
        self.cal_wl = cal_wl
        self.cal_cong = cal_cong

        self.rows = max(1, int(benchmark.grid_rows))
        self.cols = max(1, int(benchmark.grid_cols))
        self.gw = self.W / self.cols
        self.gh = self.H / self.rows
        self.bin_area = self.gw * self.gh
        self.num_bins = self.rows * self.cols

        hrm = float(benchmark.hroutes_per_micron)
        vrm = float(benchmark.vroutes_per_micron)
        self.grid_h_routes = max(1e-9, self.gh * hrm)
        self.grid_v_routes = max(1e-9, self.gw * vrm)
        self.smooth_range = (_IBM_SMOOTH_RANGE if smooth_range is None
                             else int(smooth_range))
        self.h_alloc = (hrm * _H_ALLOC_RATIO if hrouting_alloc is None
                        else float(hrouting_alloc))
        self.v_alloc = (vrm * _V_ALLOC_RATIO if vrouting_alloc is None
                        else float(vrouting_alloc))

        self.sizes = benchmark.macro_sizes.float()
        self.macro_fixed = benchmark.macro_fixed.bool()

        # net / pin structure (driver = first pin of each net)
        self.owner, self.offx, self.offy, self.net_ptr = _net_pin_lists(benchmark)
        self.net_w = benchmark.net_weights.float().numpy().astype(np.float64)
        nports = benchmark.port_positions.shape[0]
        self.ports = (benchmark.port_positions.float().numpy()
                      if nports > 0 else np.zeros((0, 2)))

        # torch copies for the vectorized wirelength/density terms
        self._t_owner = torch.from_numpy(self.owner)
        self._t_off = torch.tensor(
            np.stack([self.offx, self.offy], axis=1), dtype=torch.float32)
        self._t_netid = torch.repeat_interleave(
            torch.arange(self.num_nets),
            torch.from_numpy(np.diff(self.net_ptr)))
        self._t_netw = benchmark.net_weights.float()
        self._t_ports = (benchmark.port_positions.float()
                         if nports > 0 else torch.zeros(0, 2))
        self._t_sizes = self.sizes

    # ---- wirelength & density (vectorized torch) ----------------------

    def _wirelength(self, placement):
        all_pos = torch.cat([placement.float(), self._t_ports], dim=0)
        pin = all_pos[self._t_owner] + self._t_off
        nseg = self.num_nets
        out_max = torch.full((nseg, 2), -1e30)
        out_min = torch.full((nseg, 2), 1e30)
        idx = self._t_netid.unsqueeze(1).expand(-1, 2)
        hi = out_max.scatter_reduce(0, idx, pin, reduce="amax", include_self=True)
        lo = out_min.scatter_reduce(0, idx, pin, reduce="amin", include_self=True)
        span = (hi - lo).sum(dim=1)
        span = torch.where(hi[:, 0] > -1e29, span, torch.zeros_like(span))
        total = float((self._t_netw * span).sum())
        return self.cal_wl * total / ((self.W + self.H) * max(1, self.num_nets))

    def _density(self, placement):
        pos = placement.float()
        half = self._t_sizes / 2.0
        cols = torch.arange(self.cols).float()
        rows = torch.arange(self.rows).float()
        bx_lo, bx_hi = cols * self.gw, (cols + 1) * self.gw
        by_lo, by_hi = rows * self.gh, (rows + 1) * self.gh
        ox = torch.clamp(torch.minimum((pos[:, 0] + half[:, 0]).unsqueeze(1), bx_hi)
                         - torch.maximum((pos[:, 0] - half[:, 0]).unsqueeze(1),
                                         bx_lo), min=0.0)
        oy = torch.clamp(torch.minimum((pos[:, 1] + half[:, 1]).unsqueeze(1), by_hi)
                         - torch.maximum((pos[:, 1] - half[:, 1]).unsqueeze(1),
                                         by_lo), min=0.0)
        occ = torch.einsum("nr,nc->rc", oy, ox)
        density = (occ / self.bin_area).flatten()
        nz = density[density != 0.0]
        if nz.numel() == 0:
            return 0.0
        if self.num_bins < 10:
            return 0.5 * float(nz.mean())
        k = max(1, int(math.floor(0.1 * self.num_bins)))
        top = torch.topk(nz, min(k, nz.numel())).values
        return 0.5 * float(top.sum()) / k

    # ---- congestion (faithful numpy routing) --------------------------

    def _congestion(self, placement):
        pl = placement.detach().cpu().numpy().astype(np.float64)
        all_pos = np.concatenate([pl, self.ports], axis=0)
        px = all_pos[self.owner, 0] + self.offx
        py = all_pos[self.owner, 1] + self.offy
        pcol = np.clip((px / self.gw).astype(np.int64), 0, self.cols - 1)
        prow = np.clip((py / self.gh).astype(np.int64), 0, self.rows - 1)

        gc = self.cols
        H = np.zeros((self.rows, self.cols))
        V = np.zeros((self.rows, self.cols))
        for i in range(self.num_nets):
            s, e = self.net_ptr[i], self.net_ptr[i + 1]
            if e - s < 2:
                continue
            rr, cc = prow[s:e], pcol[s:e]
            cells = list({(int(r), int(c)) for r, c in zip(rr, cc)})
            w = float(self.net_w[i]) if i < len(self.net_w) else 1.0
            source = (int(rr[0]), int(cc[0]))
            n = len(cells)
            if n == 2:
                _two_pin(H, V, source, cells[0] if cells[1] == source
                         else cells[1], w)
            elif n == 3:
                _three_pin(H, V, cells, w)
            elif n > 3:
                for c in cells:
                    if c != source:
                        _two_pin(H, V, source, c, w)

        Hm, Vm = self._macro_routing(pl)
        H /= self.grid_h_routes
        V /= self.grid_v_routes
        Hm /= self.grid_h_routes
        Vm /= self.grid_v_routes
        H = self._smooth(H, axis="h")
        V = self._smooth(V, axis="v")
        H += Hm
        V += Vm

        allc = np.concatenate([V.ravel(), H.ravel()])
        cnt = int(math.floor(len(allc) * 0.05))
        if cnt == 0:
            return float(allc.max())
        part = np.partition(allc, -cnt)[-cnt:]
        return float(part.sum() / cnt)

    def _macro_routing(self, pl):
        """Per-hard-macro routing demand (port of __macro_route_over_grid_cell)."""
        Hm = np.zeros((self.rows, self.cols))
        Vm = np.zeros((self.rows, self.cols))
        for m in range(self.num_hard):
            mx, my = pl[m, 0], pl[m, 1]
            mw = float(self.sizes[m, 0])
            mh = float(self.sizes[m, 1])
            x_min, x_max = mx - mw / 2, mx + mw / 2
            y_min, y_max = my - mh / 2, my + mh / 2
            bl_col = int(np.clip(math.floor((mx - mw / 2) / self.gw), 0, self.cols - 1))
            ur_col = int(np.clip(math.floor((mx + mw / 2) / self.gw), 0, self.cols - 1))
            bl_row = int(np.clip(math.floor((my - mh / 2) / self.gh), 0, self.rows - 1))
            ur_row = int(np.clip(math.floor((my + mh / 2) / self.gh), 0, self.rows - 1))
            part_v = part_h = False
            for r in range(bl_row, ur_row + 1):
                for c in range(bl_col, ur_col + 1):
                    cx0, cx1 = c * self.gw, (c + 1) * self.gw
                    cy0, cy1 = r * self.gh, (r + 1) * self.gh
                    xd = min(x_max, cx1) - max(x_min, cx0)
                    yd = min(y_max, cy1) - max(y_min, cy0)
                    if not (xd > 0 and yd > 0):
                        xd = yd = 0.0
                    if ur_row != bl_row and r in (bl_row, ur_row) \
                            and abs(yd - self.gh) > 1e-5:
                        part_v = True
                    if ur_col != bl_col and c in (bl_col, ur_col) \
                            and abs(xd - self.gw) > 1e-5:
                        part_h = True
                    Vm[r, c] += xd * self.v_alloc
                    Hm[r, c] += yd * self.h_alloc
            if part_v:
                for c in range(bl_col, ur_col + 1):
                    cx0, cx1 = c * self.gw, (c + 1) * self.gw
                    cy0, cy1 = ur_row * self.gh, (ur_row + 1) * self.gh
                    xd = min(x_max, cx1) - max(x_min, cx0)
                    yd = min(y_max, cy1) - max(y_min, cy0)
                    if xd > 0 and yd > 0:
                        Vm[ur_row, c] -= xd * self.v_alloc
            if part_h:
                for r in range(bl_row, ur_row + 1):
                    cx0, cx1 = ur_col * self.gw, (ur_col + 1) * self.gw
                    cy0, cy1 = r * self.gh, (r + 1) * self.gh
                    xd = min(x_max, cx1) - max(x_min, cx0)
                    yd = min(y_max, cy1) - max(y_min, cy0)
                    if xd > 0 and yd > 0:
                        Hm[r, ur_col] -= yd * self.h_alloc
        return Hm, Vm

    def _smooth(self, grid, axis):
        """Spreading smooth (port of __smooth_routing_cong)."""
        r = self.smooth_range
        if r <= 0:
            return grid
        out = np.zeros_like(grid)
        if axis == "v":  # spread along columns
            for col in range(self.cols):
                lp = max(0, col - r)
                rp = min(self.cols - 1, col + r)
                out[:, lp:rp + 1] += (grid[:, col] / (rp - lp + 1))[:, None]
        else:  # spread along rows
            for row in range(self.rows):
                lp = max(0, row - r)
                up = min(self.rows - 1, row + r)
                out[lp:up + 1, :] += (grid[row, :] / (up - lp + 1))[None, :]
        return out

    # ---- public API ---------------------------------------------------

    def __call__(self, placement):
        """Return a dict with proxy_cost and the three component costs."""
        wl = self._wirelength(placement)
        den = self._density(placement)
        cong = self.cal_cong * self._congestion(placement)
        return {
            "proxy_cost": wl + 0.5 * den + 0.5 * cong,
            "wirelength_cost": wl,
            "density_cost": den,
            "congestion_cost": cong,
        }


def _two_pin(H, V, source, sink, w):
    sr, sc = source
    kr, kc = sink
    H[sr, min(sc, kc):max(sc, kc)] += w
    V[min(sr, kr):max(sr, kr), kc] += w


def _l_routing(H, V, cells, w):
    cs = sorted(cells, key=lambda p: (p[1], p[0]))
    (y1, x1), (y2, x2), (y3, x3) = cs
    H[y1, x1:x2] += w
    H[y2, x2:x3] += w
    V[min(y1, y2):max(y1, y2), x2] += w
    V[min(y2, y3):max(y2, y3), x3] += w


def _t_routing(H, V, cells, w):
    cs = sorted(cells)
    (y1, x1), (y2, x2), (y3, x3) = cs
    xmin, xmax = min(x1, x2, x3), max(x1, x2, x3)
    H[y2, xmin:xmax] += w
    V[min(y1, y2):max(y1, y2), x1] += w
    V[min(y2, y3):max(y2, y3), x3] += w


def _three_pin(H, V, cells, w):
    cs = sorted(cells, key=lambda p: (p[1], p[0]))
    (y1, x1), (y2, x2), (y3, x3) = cs
    if x1 < x2 < x3 and min(y1, y3) < y2 and max(y1, y3) > y2:
        _l_routing(H, V, cs, w)
    elif x2 == x3 and x1 < x2 and y1 < min(y2, y3):
        H[y1, x1:x2] += w
        V[y1:max(y2, y3), x2] += w
    elif y2 == y3:
        H[y1, x1:x2] += w
        H[y2, x2:x3] += w
        V[min(y2, y1):max(y2, y1), x2] += w
    else:
        _t_routing(H, V, cs, w)
