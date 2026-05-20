"""Analytical placer v2: weighted-average wirelength + Nesterov SGD.

Same overall structure as :class:`AnalyticalPlacer` (electrostatic-style joint
optimisation, push-apart legaliser, shelf-pack fallback) but replaces two
components that DREAMPlace and ePlace identify as the dominant determinants
of analytical-placement quality:

1. **Weighted-average wirelength (WA-WL)** instead of log-sum-exp HPWL. LSE
   concentrates the gradient on the single most-extreme pin per net; WA-WL
   distributes it across all pins weighted by ``exp(x / gamma)``. The wider
   gradient support converges faster and reaches better local minima on
   designs with many high-fanout nets. Reference: ePlace ToDAES'15,
   DREAMPlace ``weighted_average_wirelength`` op.

2. **Nesterov-momentum SGD** instead of Adam. ePlace and DREAMPlace use
   Nesterov for placement; Adam's per-parameter adaptive learning rates
   interact poorly with the spreading dynamics (macros that already escaped
   their crowded bin keep accelerating, causing oscillation). Nesterov with
   ``momentum=0.9`` and a single learning rate produces a smoother path to
   convergence.

The objective signal flow, density loss, overlap barrier, deadline handling,
and legaliser are identical to ``AnalyticalPlacer``. The shared code lives in
``engine.py``; this module overrides only ``_hpwl_loss`` (WA-WL) and the
optimiser construction.
"""

import time

import torch

from .engine import (
    AnalyticalPlacer,
    _build_pin_index,
    _seg_amax,
    _seg_sum,
    clearance_microns,
)


class AnalyticalPlacerV2(AnalyticalPlacer):
    """WA-WL + Nesterov variant of :class:`AnalyticalPlacer`.

    Hyperparameter defaults are re-tuned for Nesterov: ``lr`` is higher than
    the Adam defaults (Nesterov needs larger step sizes), and
    ``overlap_weight`` is unchanged (the overlap barrier scaling is
    invariant to the optimiser choice).
    """

    def __init__(
        self,
        n_iters=400,
        lr=0.05,
        momentum=0.9,
        gamma_start=0.015,
        gamma_end=0.002,
        wl_weight=1.0,
        density_weight=0.5,
        overlap_weight=6.0,
        legalize_gap=0.003,
        density_topk_frac=1.0,
        congestion_weight=0.0,
        congestion_topk_frac=0.05,
        seed=0,
        device="cpu",
        verbose=False,
    ):
        super().__init__(
            n_iters=n_iters,
            lr=lr,
            gamma_start=gamma_start,
            gamma_end=gamma_end,
            wl_weight=wl_weight,
            density_weight=density_weight,
            overlap_weight=overlap_weight,
            legalize_gap=legalize_gap,
            density_topk_frac=density_topk_frac,
            congestion_weight=congestion_weight,
            congestion_topk_frac=congestion_topk_frac,
            seed=seed,
            device=device,
            verbose=verbose,
        )
        self.momentum = momentum

    @staticmethod
    def _wa_wl_loss(pin_pos, netid, num_nets, net_w, gamma):
        """Weighted-average wirelength.

        Per dimension :math:`d \\in \\{x, y\\}`:

        .. math::

            x_{\\max} \\approx \\frac{\\sum_i x_i e^{x_i/\\gamma}}
                                     {\\sum_i e^{x_i/\\gamma}}

            x_{\\min} \\approx \\frac{\\sum_i x_i e^{-x_i/\\gamma}}
                                     {\\sum_i e^{-x_i/\\gamma}}

        and the dimension contribution is :math:`x_{\\max} - x_{\\min}`. The
        log-sum-exp variant in :meth:`AnalyticalPlacer._hpwl_loss` would
        replace the weighted averages with smooth-max/min via
        :math:`\\gamma \\log \\sum e^{x_i/\\gamma}`; WA-WL distributes the
        gradient across all pins instead of concentrating it on the extreme.

        Numerically stabilised via per-segment max subtraction inside the
        exponentials (same trick the LSE path uses).
        """
        total = torch.zeros((), device=pin_pos.device)
        for d in (0, 1):
            x = pin_pos[:, d]
            # max side: weights ~ exp(x/gamma)
            z = x / gamma
            zmax = _seg_amax(z, netid, num_nets)
            ez = torch.exp(z - zmax[netid])  # per-pin, numerically stable
            sum_ez = _seg_sum(ez, netid, num_nets)  # per-net partition
            sum_xez = _seg_sum(x * ez, netid, num_nets)
            wa_max = sum_xez / (sum_ez + 1e-12)
            # min side: weights ~ exp(-x/gamma)
            zn = -x / gamma
            znmax = _seg_amax(zn, netid, num_nets)
            ezn = torch.exp(zn - znmax[netid])
            sum_ezn = _seg_sum(ezn, netid, num_nets)
            sum_xezn = _seg_sum(x * ezn, netid, num_nets)
            wa_min = sum_xezn / (sum_ezn + 1e-12)
            total = total + (net_w * (wa_max - wa_min)).sum()
        return total / max(1, num_nets)

    def place(self, benchmark, deadline=None):
        """Same as :meth:`AnalyticalPlacer.place` but with WA-WL + Nesterov.

        Copied (not refactored) from the parent so the inner loop stays a
        single flat block -- the parent's ``place`` is not split into
        overridable steps and pulling the loop out for one-method reuse
        would add more complexity than it removes.
        """
        device = torch.device(self.device)
        torch.manual_seed(self.seed)
        if device.type == "cuda":
            try:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
                torch.backends.cuda.matmul.allow_tf32 = False
                torch.backends.cudnn.allow_tf32 = False
                import os as _os
                _os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
                torch.use_deterministic_algorithms(True, warn_only=True)
            except Exception:
                pass

        N = benchmark.num_macros
        nh = benchmark.num_hard_macros
        cw = float(benchmark.canvas_width)
        ch = float(benchmark.canvas_height)
        canvas = torch.tensor([cw, ch], dtype=torch.float32, device=device)

        sizes = benchmark.macro_sizes.to(device).float()
        sizes_n = sizes / canvas
        half = sizes_n / 2.0
        clr_um = clearance_microns(cw, ch)
        g_push = max(self.legalize_gap, clr_um / cw, clr_um / ch)
        lo = torch.minimum(half, torch.full_like(half, 0.5))
        hi = torch.maximum(1.0 - half, lo)
        fixed = benchmark.macro_fixed.to(device).bool()

        ref = benchmark.macro_positions.to(device).float()
        init = torch.clamp(ref / canvas, lo, hi)

        var = init.clone().detach().requires_grad_(True)
        # Nesterov SGD instead of Adam. lr is calibrated higher to compensate
        # for the lack of Adam's per-parameter adaptive scaling.
        opt = torch.optim.SGD([var], lr=self.lr, momentum=self.momentum,
                              nesterov=True)

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
                # OVERRIDE: WA-WL instead of LSE-HPWL.
                wl = self._wa_wl_loss(
                    pin_pos, p_netid, num_nets, net_w, gamma)
                loss = loss + self.wl_weight * wl

            dens = self._density_loss(
                var, half, bx_lo, bx_hi, by_lo, by_hi, bin_area,
                density_target, self.density_topk_frac,
            )
            loss = loss + self.density_weight * dens

            if self.congestion_weight > 0 and has_nets:
                cong = self._congestion_loss(
                    pin_pos, p_netid, num_nets, net_w,
                    bx_lo, bx_hi, by_lo, by_hi, bin_area,
                    self.congestion_topk_frac,
                )
                loss = loss + self.congestion_weight * cong

            if tri is not None and ov_w > 0:
                ovl = self._overlap_loss(var, half, nh, tri)
                loss = loss + ov_w * ovl

            loss.backward()
            opt.step()

            with torch.no_grad():
                var.clamp_(lo, hi)
                var[fixed] = init[fixed]

            if self.verbose and (it % 100 == 0 or it == self.n_iters - 1):
                print(f"  [v2] iter {it:4d}  loss={loss.item():.5f}  "
                      f"gamma={gamma:.4f}")

        pos = var.detach().clone()
        pos[fixed] = init[fixed]

        # Legalise with the parent's push-apart + shelf-pack chain.
        pos, remaining = self._legalize_hard(
            pos, half, fixed, nh, lo, hi, g_push, deadline=deadline)
        if remaining > 0:
            if self.verbose:
                print(f"  [v2] {remaining} overlaps remain -> shelf-pack")
            shelf_gap = float(max(clr_um / cw, clr_um / ch))
            pos = self._shelf_pack(pos, sizes_n, fixed, nh, shelf_gap)

        out = pos * canvas
        out = torch.clamp(out, sizes / 2.0, canvas - sizes / 2.0)
        out[fixed] = ref[fixed]
        return out.cpu()
