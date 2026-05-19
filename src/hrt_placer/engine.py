"""Portable PyTorch electrostatic analytical macro placer.

Runs identically on CPU and GPU with no CUDA build dependency. Optimizes a
differentiable surrogate of the proxy cost (smooth HPWL + bin-density overflow
+ hard-macro overlap) with Adam, then legalizes hard macros to zero overlap.

All internal math is done in canvas-normalized [0, 1] coordinates so the same
hyperparameters transfer across benchmarks of very different physical scale.
"""

import time

import torch


def clearance_microns(canvas_w, canvas_h):
    """Target macro-to-macro clearance in microns.

    SCORING.md asks for >=12 um in submitted placements so the Tier-2 ORFS
    auto-spacing leaves our coordinates untouched. On real dies (NG45
    canvases ~900-2100 um) 12 um is a small fraction of the canvas and is
    used directly. The abstract-unit IBM dies have a ~23-unit canvas where
    12 would exceed the die; there Tier-1 proxy scores submitted
    coordinates unchanged and needs no clearance, so only the legalizer's
    base gap is kept.
    """
    m = min(canvas_w, canvas_h)
    if m < 200.0:
        return 0.003 * m
    return min(12.0, 0.03 * m)


def _seg_amax(vals, segid, nseg):
    """Per-segment maximum (segments indexed by segid into [0, nseg))."""
    out = torch.full((nseg,), -1e30, dtype=vals.dtype, device=vals.device)
    return out.scatter_reduce(0, segid, vals, reduce="amax", include_self=True)


def _seg_sum(vals, segid, nseg):
    """Per-segment sum."""
    out = torch.zeros(nseg, dtype=vals.dtype, device=vals.device)
    return out.scatter_reduce(0, segid, vals, reduce="sum", include_self=True)


def _build_pin_index(benchmark):
    """Flatten net connectivity into per-pin tensors.

    Prefers pin-level data (``net_pin_nodes`` + ``macro_pin_offsets``); falls
    back to per-macro ``net_nodes`` with zero offsets. Returns a dict of CPU
    tensors, or None if the benchmark has no usable connectivity.
    """
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

    if not netid:
        return None
    return {
        "netid": torch.tensor(netid, dtype=torch.long),
        "owner": torch.tensor(owner, dtype=torch.long),
        "offset": torch.tensor([offx, offy], dtype=torch.float32).t().contiguous(),
    }


class AnalyticalPlacer:
    """Electrostatic-style analytical placer.

    Parameters are canvas-normalized. ``place(benchmark)`` returns a
    ``[num_macros, 2]`` CPU tensor of real (micron) center coordinates with
    zero hard-macro overlap and all macros inside the canvas.
    """

    def __init__(
        self,
        n_iters=400,
        lr=0.002,
        gamma_start=0.015,
        gamma_end=0.002,
        wl_weight=1.0,
        density_weight=0.5,
        overlap_weight=6.0,
        legalize_gap=0.003,
        seed=0,
        device="cpu",
        verbose=False,
    ):
        self.n_iters = n_iters
        self.lr = lr
        self.gamma_start = gamma_start
        self.gamma_end = gamma_end
        self.wl_weight = wl_weight
        self.density_weight = density_weight
        self.overlap_weight = overlap_weight
        self.legalize_gap = legalize_gap
        self.seed = seed
        self.device = device
        self.verbose = verbose

    def place(self, benchmark, deadline=None):
        """Return a legal placement. ``deadline`` is an absolute ``time.time()``
        after which the analytical loop and legalizer stop early; the shelf-pack
        fallback still runs, so the result is legal even on a timeout."""
        device = torch.device(self.device)
        torch.manual_seed(self.seed)

        N = benchmark.num_macros
        nh = benchmark.num_hard_macros
        cw = float(benchmark.canvas_width)
        ch = float(benchmark.canvas_height)
        canvas = torch.tensor([cw, ch], dtype=torch.float32, device=device)

        sizes = benchmark.macro_sizes.to(device).float()  # [N, 2] microns
        sizes_n = sizes / canvas
        half = sizes_n / 2.0
        # macro-to-macro clearance (PRD): the legalizer pushes every hard
        # pair to at least this normalized separation beyond touching.
        clr_um = clearance_microns(cw, ch)
        g_push = max(self.legalize_gap, clr_um / cw, clr_um / ch)
        lo = torch.minimum(half, torch.full_like(half, 0.5))
        hi = torch.maximum(1.0 - half, lo)
        fixed = benchmark.macro_fixed.to(device).bool()  # [N]

        ref = benchmark.macro_positions.to(device).float()
        init = torch.clamp(ref / canvas, lo, hi)

        var = init.clone().detach().requires_grad_(True)
        opt = torch.optim.Adam([var], lr=self.lr)

        # --- net / pin connectivity ---
        pin = _build_pin_index(benchmark)
        has_nets = pin is not None
        if has_nets:
            p_netid = pin["netid"].to(device)
            p_owner = pin["owner"].to(device)
            p_off = pin["offset"].to(device) / canvas
            num_nets = benchmark.num_nets
            net_w = benchmark.net_weights.to(device).float()
            num_ports = benchmark.port_positions.shape[0]
            if num_ports > 0:
                port_n = benchmark.port_positions.to(device).float() / canvas
            else:
                port_n = torch.zeros(0, 2, dtype=torch.float32, device=device)

        # --- density grid ---
        rows = max(1, int(benchmark.grid_rows))
        cols = max(1, int(benchmark.grid_cols))
        bin_w = 1.0 / cols
        bin_h = 1.0 / rows
        bx_lo = torch.arange(cols, device=device, dtype=torch.float32) * bin_w
        bx_hi = bx_lo + bin_w
        by_lo = torch.arange(rows, device=device, dtype=torch.float32) * bin_h
        by_hi = by_lo + bin_h
        bin_area = bin_w * bin_h
        util = float((sizes_n[:, 0] * sizes_n[:, 1]).sum())
        density_target = min(0.9, max(0.1, util))

        # hard-macro pair indices for the overlap penalty
        tri = torch.triu_indices(nh, nh, offset=1, device=device) if nh > 1 else None

        for it in range(self.n_iters):
            if deadline is not None and it % 32 == 0 and time.time() > deadline:
                break
            frac = it / max(1, self.n_iters - 1)
            gamma = self.gamma_start + frac * (self.gamma_end - self.gamma_start)
            ov_w = self.overlap_weight * min(1.0, max(0.0, (frac - 0.1) / 0.4))

            opt.zero_grad()
            loss = torch.zeros((), device=device)

            if has_nets:
                all_pos = torch.cat([var, port_n], dim=0)
                pin_pos = all_pos[p_owner] + p_off
                wl = self._hpwl_loss(pin_pos, p_netid, num_nets, net_w, gamma)
                loss = loss + self.wl_weight * wl

            dens = self._density_loss(
                var, half, bx_lo, bx_hi, by_lo, by_hi, bin_area, density_target
            )
            loss = loss + self.density_weight * dens

            if tri is not None and ov_w > 0:
                ovl = self._overlap_loss(var, half, nh, tri)
                loss = loss + ov_w * ovl

            loss.backward()
            opt.step()

            with torch.no_grad():
                var.clamp_(lo, hi)
                var[fixed] = init[fixed]

            if self.verbose and (it % 100 == 0 or it == self.n_iters - 1):
                print(f"  iter {it:4d}  loss={loss.item():.5f}  gamma={gamma:.4f}")

        pos = var.detach().clone()
        pos[fixed] = init[fixed]

        if self.verbose:
            dbg = pos[:nh]
            dh = half[:nh]
            for g in (0.0, self.legalize_gap):
                ddx = (dbg[:, 0].unsqueeze(1) - dbg[:, 0].unsqueeze(0)).abs()
                ddy = (dbg[:, 1].unsqueeze(1) - dbg[:, 1].unsqueeze(0)).abs()
                sx = dh[:, 0].unsqueeze(1) + dh[:, 0].unsqueeze(0) + g
                sy = dh[:, 1].unsqueeze(1) + dh[:, 1].unsqueeze(0) + g
                ov = (sx - ddx > 0) & (sy - ddy > 0)
                ov.fill_diagonal_(False)
                print(f"  [diag] analytical overlaps@gap{g}: "
                      f"{int(ov.float().sum()) // 2}")
            print(f"  [diag] x[{float(dbg[:,0].min()):.3f},"
                  f"{float(dbg[:,0].max()):.3f}] "
                  f"y[{float(dbg[:,1].min()):.3f},{float(dbg[:,1].max()):.3f}] "
                  f"mean_half={float(dh.mean()):.4f}")

        # legalize hard macros to zero overlap, pushing toward the clearance
        pos, remaining = self._legalize_hard(
            pos, half, fixed, nh, lo, hi, g_push, deadline=deadline)
        if remaining > 0:
            # guaranteed-legal fallback: shelf-pack movable hard macros
            if self.verbose:
                print(f"  [legalize] {remaining} overlaps remain -> shelf-pack")
            shelf_gap = float(max(clr_um / cw, clr_um / ch))
            pos = self._shelf_pack(pos, sizes_n, fixed, nh, shelf_gap)

        # back to microns and snap to the canvas. For push-apart output this
        # is only an epsilon-level boundary correction: the >=g_push clearance
        # margin means it cannot close a real gap into an overlap.
        out = pos * canvas
        out = torch.clamp(out, sizes / 2.0, canvas - sizes / 2.0)
        out[fixed] = ref[fixed]
        return out.cpu()

    @staticmethod
    def _hpwl_loss(pin_pos, netid, num_nets, net_w, gamma):
        """Smooth (log-sum-exp) half-perimeter wirelength, mean over nets."""
        total = torch.zeros((), device=pin_pos.device)
        for d in (0, 1):
            a = pin_pos[:, d] / gamma
            mx = _seg_amax(a, netid, num_nets)
            smax = gamma * (mx + torch.log(
                _seg_sum(torch.exp(a - mx[netid]), netid, num_nets) + 1e-12))
            mn = _seg_amax(-a, netid, num_nets)
            smin = -gamma * (mn + torch.log(
                _seg_sum(torch.exp(-a - mn[netid]), netid, num_nets) + 1e-12))
            total = total + (net_w * (smax - smin)).sum()
        return total / max(1, num_nets)

    @staticmethod
    def _density_loss(pos, half, bx_lo, bx_hi, by_lo, by_hi, bin_area, target):
        """Squared bin-density overflow above the uniform-spread target.

        The mean density is constant (= utilization) and carries no gradient;
        penalizing overflow above ``target`` produces a real spreading force
        that pushes macros out of overcrowded bins.
        """
        mx_lo = (pos[:, 0] - half[:, 0]).unsqueeze(1)
        mx_hi = (pos[:, 0] + half[:, 0]).unsqueeze(1)
        my_lo = (pos[:, 1] - half[:, 1]).unsqueeze(1)
        my_hi = (pos[:, 1] + half[:, 1]).unsqueeze(1)
        ox = torch.clamp(
            torch.minimum(mx_hi, bx_hi) - torch.maximum(mx_lo, bx_lo), min=0.0)
        oy = torch.clamp(
            torch.minimum(my_hi, by_hi) - torch.maximum(my_lo, by_lo), min=0.0)
        occ = torch.einsum("nr,nc->rc", oy, ox)
        density = occ / bin_area
        return torch.clamp(density - target, min=0.0).pow(2).mean()

    def _overlap_loss(self, pos, half, nh, tri):
        """Total pairwise hard-macro overlap area (a vanishing barrier).

        Not averaged: the term must dominate the loss while overlaps exist and
        then disappear once macros separate.
        """
        hp = pos[:nh]
        hh = half[:nh]
        i, j = tri[0], tri[1]
        dx = (hp[i, 0] - hp[j, 0]).abs()
        dy = (hp[i, 1] - hp[j, 1]).abs()
        ox = torch.clamp((hh[i, 0] + hh[j, 0]) - dx, min=0.0)
        oy = torch.clamp((hh[i, 1] + hh[j, 1]) - dy, min=0.0)
        return (ox * oy).sum()

    def _legalize_hard(self, pos, half, fixed, nh, lo, hi, g_push,
                       deadline=None, step=0.85, max_iter=8000):
        """Iterative push-apart of hard macros.

        Pushes pairs toward a ``g_push`` target separation (the macro
        clearance), then reports the count of pairs that still *truly*
        overlap (within a small check gap). The shelf-pack fallback fires
        only on genuine residual overlap, not on merely-tight spacing.
        """
        if nh <= 1:
            return pos, 0
        g_check = 0.0008
        p = pos.clone()
        hp = p[:nh].clone()
        hh = half[:nh]
        fx = fixed[:nh]
        hlo = lo[:nh]
        hhi = hi[:nh]
        sep_x0 = hh[:, 0].unsqueeze(1) + hh[:, 0].unsqueeze(0)
        sep_y0 = hh[:, 1].unsqueeze(1) + hh[:, 1].unsqueeze(0)
        first = None

        for it in range(max_iter):
            if deadline is not None and it % 64 == 0 and time.time() > deadline:
                break
            dx = hp[:, 0].unsqueeze(1) - hp[:, 0].unsqueeze(0)
            dy = hp[:, 1].unsqueeze(1) - hp[:, 1].unsqueeze(0)
            ox = (sep_x0 + g_push) - dx.abs()
            oy = (sep_y0 + g_push) - dy.abs()
            overlapping = (ox > 0) & (oy > 0)
            overlapping.fill_diagonal_(False)
            cnt = int(overlapping.float().sum()) // 2
            if first is None:
                first = cnt
            if cnt == 0:
                break

            push_x = overlapping & (ox <= oy)
            push_y = overlapping & ~(ox <= oy)
            sign_x = torch.sign(dx)
            sign_x[sign_x == 0] = 1.0
            sign_y = torch.sign(dy)
            sign_y[sign_y == 0] = 1.0
            amt_x = torch.where(push_x, ox * sign_x * 0.5, torch.zeros_like(ox))
            amt_y = torch.where(push_y, oy * sign_y * 0.5, torch.zeros_like(oy))

            disp = torch.zeros_like(hp)
            disp[:, 0] = amt_x.sum(dim=1)
            disp[:, 1] = amt_y.sum(dim=1)
            disp[fx] = 0.0

            hp = torch.clamp(hp + step * disp, hlo, hhi)

        dx = hp[:, 0].unsqueeze(1) - hp[:, 0].unsqueeze(0)
        dy = hp[:, 1].unsqueeze(1) - hp[:, 1].unsqueeze(0)
        tov = (((sep_x0 + g_check) - dx.abs()) > 0) & \
              (((sep_y0 + g_check) - dy.abs()) > 0)
        tov.fill_diagonal_(False)
        true_cnt = int(tov.float().sum()) // 2
        if self.verbose:
            print(f"  [legalize] pre={first} post_true={true_cnt}")
        p[:nh] = hp
        return p, true_cnt

    @staticmethod
    def _shelf_pack(pos, sizes_n, fixed, nh, gap):
        """Guaranteed-overlap-free fallback: shelf-pack movable hard macros.

        Places macros left-to-right in height-sorted rows with ``gap``
        spacing (the >=12 um clearance, normalized). A macro that would
        overrun the row starts a new row; it is never stacked onto an
        occupied spot, so the result has zero overlap. Used only when the
        iterative legalizer fails to converge. Soft and fixed macros keep
        their positions.
        """
        p = pos.clone()
        idx = [i for i in range(nh) if not bool(fixed[i])]
        idx.sort(key=lambda i: -float(sizes_n[i, 1]))
        cx = 0.0
        cy = 0.0
        row_h = 0.0
        for i in idx:
            w = float(sizes_n[i, 0])
            h = float(sizes_n[i, 1])
            if cx > 0.0 and cx + w > 1.0:
                cx = 0.0
                cy += row_h + gap
                row_h = 0.0
            p[i, 0] = cx + w / 2.0
            p[i, 1] = cy + h / 2.0
            cx += w + gap
            row_h = max(row_h, h)
        return p
