"""From-Benchmark reimplementation of the TILOS proxy cost.

The challenge evaluator (``macro_place.objective.compute_proxy_cost``) needs the
``PlacementCost`` object, which ``place()`` never receives. To score and refine
candidates at runtime — and on unseen Tier 2 designs — the placer needs its own
proxy-cost model built purely from the ``Benchmark`` tensors.

Wirelength and density are reproduced closely; congestion uses a RUDY-style
routing-demand estimate (per-net bounding-box demand spread over the grid).
``ProxyCost`` precomputes the net/grid structure once per benchmark; ``__call__``
then scores a placement. All math is torch, so the same code is reusable for a
differentiable objective later.
"""

import math

import torch


def build_pins(benchmark):
    """Flatten net connectivity into per-pin tensors (netid, owner, offset)."""
    num_hard = benchmark.num_hard_macros
    netid, owner, offx, offy = [], [], [], []
    use_pin = (
        len(benchmark.net_pin_nodes) == benchmark.num_nets
        and benchmark.num_nets > 0
    )
    if use_pin:
        for nid, pins in enumerate(benchmark.net_pin_nodes):
            if pins.numel() == 0:
                continue
            for o, pidx in pins.tolist():
                netid.append(nid)
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
        for nid, nodes in enumerate(benchmark.net_nodes):
            for o in nodes.tolist():
                netid.append(nid)
                owner.append(o)
                offx.append(0.0)
                offy.append(0.0)
    return (
        torch.tensor(netid, dtype=torch.long),
        torch.tensor(owner, dtype=torch.long),
        torch.tensor([offx, offy], dtype=torch.float32).t().contiguous(),
    )


class ProxyCost:
    """Proxy-cost model reconstructed from a ``Benchmark``.

    ``cost = 1.0 * wirelength + 0.5 * density + 0.5 * congestion``.
    """

    def __init__(self, benchmark, device="cpu", smooth_range=2,
                 cal_wl=0.84, cal_den=0.5, cal_cong=1.5):
        self.device = torch.device(device)
        # Calibration scales fitted against the real TILOS evaluator on IBM
        # benchmarks (density is exactly 2x, wirelength ~1.19x; congestion is
        # an approximate RUDY model and the least faithful term).
        self.cal_wl = cal_wl
        self.cal_den = cal_den
        self.cal_cong = cal_cong
        self.W = float(benchmark.canvas_width)
        self.H = float(benchmark.canvas_height)
        self.num_macros = benchmark.num_macros
        self.num_nets = benchmark.num_nets
        self.smooth_range = smooth_range

        self.sizes = benchmark.macro_sizes.to(self.device).float()
        self.rows = max(1, int(benchmark.grid_rows))
        self.cols = max(1, int(benchmark.grid_cols))
        self.bin_w = self.W / self.cols
        self.bin_h = self.H / self.rows
        self.bin_area = self.bin_w * self.bin_h
        self.num_bins = self.rows * self.cols

        # bin edges (microns)
        self.bx_lo = torch.arange(self.cols, device=self.device).float() * self.bin_w
        self.bx_hi = self.bx_lo + self.bin_w
        self.by_lo = torch.arange(self.rows, device=self.device).float() * self.bin_h
        self.by_hi = self.by_lo + self.bin_h

        # routing capacity per bin (tracks crossing a bin edge)
        self.h_cap = max(1e-9, self.bin_h * float(benchmark.hroutes_per_micron))
        self.v_cap = max(1e-9, self.bin_w * float(benchmark.vroutes_per_micron))

        # net / pin structure
        netid, owner, offset = build_pins(benchmark)
        self.netid = netid.to(self.device)
        self.owner = owner.to(self.device)
        self.offset = offset.to(self.device)
        self.net_w = benchmark.net_weights.to(self.device).float()
        nports = benchmark.port_positions.shape[0]
        self.ports = (
            benchmark.port_positions.to(self.device).float()
            if nports > 0
            else torch.zeros(0, 2, device=self.device)
        )

    # ---- term helpers -------------------------------------------------

    def _seg(self, vals, reduce):
        out_init = {"amax": -1e30, "amin": 1e30, "sum": 0.0}[reduce]
        out = torch.full((self.num_nets,), out_init, dtype=vals.dtype,
                         device=vals.device)
        return out.scatter_reduce(0, self.netid, vals, reduce=reduce,
                                  include_self=True)

    def _net_bbox(self, placement):
        """Return per-net (x_lo, x_hi, y_lo, y_hi) over pin endpoints."""
        all_pos = torch.cat([placement.to(self.device), self.ports], dim=0)
        pin = all_pos[self.owner] + self.offset
        x_lo = self._seg(pin[:, 0], "amin")
        x_hi = self._seg(pin[:, 0], "amax")
        y_lo = self._seg(pin[:, 1], "amin")
        y_hi = self._seg(pin[:, 1], "amax")
        valid = x_hi > -1e29
        return x_lo, x_hi, y_lo, y_hi, valid

    def wirelength(self, x_lo, x_hi, y_lo, y_hi, valid):
        hpwl = (x_hi - x_lo) + (y_hi - y_lo)
        hpwl = torch.where(valid, hpwl, torch.zeros_like(hpwl))
        total = (self.net_w * hpwl).sum()
        return total / ((self.W + self.H) * max(1, self.num_nets))

    def density(self, placement):
        pos = placement.to(self.device)
        half = self.sizes / 2.0
        mx_lo = (pos[:, 0] - half[:, 0]).unsqueeze(1)
        mx_hi = (pos[:, 0] + half[:, 0]).unsqueeze(1)
        my_lo = (pos[:, 1] - half[:, 1]).unsqueeze(1)
        my_hi = (pos[:, 1] + half[:, 1]).unsqueeze(1)
        ox = torch.clamp(torch.minimum(mx_hi, self.bx_hi)
                         - torch.maximum(mx_lo, self.bx_lo), min=0.0)
        oy = torch.clamp(torch.minimum(my_hi, self.by_hi)
                         - torch.maximum(my_lo, self.by_lo), min=0.0)
        occ = torch.einsum("nr,nc->rc", oy, ox)
        density = (occ / self.bin_area).flatten()
        nz = density[density > 1e-9]
        if nz.numel() == 0:
            return torch.zeros((), device=self.device)
        k = max(1, int(math.floor(0.1 * self.num_bins)))
        if nz.numel() <= k:
            return nz.mean()
        return torch.topk(nz, k).values.mean()

    def _smooth_rows(self, grid, axis):
        """1-D box blur of radius ``smooth_range`` along ``axis`` (0=row,1=col)."""
        r = self.smooth_range
        if r <= 0:
            return grid
        g = grid if axis == 1 else grid.t()
        pad = torch.nn.functional.pad(g, (r, r), mode="replicate")
        kernel = torch.ones(1, 1, 2 * r + 1, device=grid.device) / (2 * r + 1)
        sm = torch.nn.functional.conv1d(
            pad.unsqueeze(1), kernel).squeeze(1)
        return sm if axis == 1 else sm.t()

    def congestion(self, x_lo, x_hi, y_lo, y_hi, valid):
        """RUDY-style routing congestion: top-5% smoothed demand/capacity."""
        w = torch.clamp(x_hi - x_lo, min=0.0)
        h = torch.clamp(y_hi - y_lo, min=0.0)
        # inflate degenerate bounding boxes to at least one bin
        wq = torch.clamp(w, min=self.bin_w)
        hq = torch.clamp(h, min=self.bin_h)
        cx = 0.5 * (x_lo + x_hi)
        cy = 0.5 * (y_lo + y_hi)
        nx_lo, nx_hi = cx - wq / 2, cx + wq / 2
        ny_lo, ny_hi = cy - hq / 2, cy + hq / 2

        ox = torch.clamp(torch.minimum(nx_hi.unsqueeze(1), self.bx_hi)
                         - torch.maximum(nx_lo.unsqueeze(1), self.bx_lo), min=0.0)
        oy = torch.clamp(torch.minimum(ny_hi.unsqueeze(1), self.by_hi)
                         - torch.maximum(ny_lo.unsqueeze(1), self.by_lo), min=0.0)
        bbox_area = (wq * hq).clamp(min=1e-9)
        vmask = valid.float()
        coef_h = self.net_w * w * vmask / bbox_area
        coef_v = self.net_w * h * vmask / bbox_area
        h_dem = torch.einsum("n,nr,nc->rc", coef_h, oy, ox)
        v_dem = torch.einsum("n,nr,nc->rc", coef_v, oy, ox)

        h_cong = self._smooth_rows(h_dem / self.h_cap, axis=1)
        v_cong = self._smooth_rows(v_dem / self.v_cap, axis=0)
        allc = torch.cat([h_cong.flatten(), v_cong.flatten()])
        k = max(1, int(math.floor(0.05 * allc.numel())))
        return torch.topk(allc, k).values.mean()

    # ---- public API ---------------------------------------------------

    def __call__(self, placement):
        """Return a dict with proxy_cost and the three component costs.

        Component costs are calibration-scaled to track the real evaluator;
        ``raw_*`` keys expose the uncalibrated values.
        """
        x_lo, x_hi, y_lo, y_hi, valid = self._net_bbox(placement)
        wl = float(self.wirelength(x_lo, x_hi, y_lo, y_hi, valid))
        den = float(self.density(placement))
        cong = float(self.congestion(x_lo, x_hi, y_lo, y_hi, valid))
        cwl, cden, ccong = (wl * self.cal_wl, den * self.cal_den,
                            cong * self.cal_cong)
        return {
            "proxy_cost": cwl + 0.5 * cden + 0.5 * ccong,
            "wirelength_cost": cwl,
            "density_cost": cden,
            "congestion_cost": ccong,
            "raw_wirelength": wl,
            "raw_density": den,
            "raw_congestion": cong,
        }
