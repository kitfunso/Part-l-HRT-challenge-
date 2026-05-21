"""Analytical placer v3: electrostatic density (eDensity) with bilinear splat.

The v1/v2 engines penalise density with a per-bin overflow term. That is a
*local* penalty -- macros in a uniformly medium-dense region feel no spreading
pressure, so the optimiser settles into clumped local minima.

ePlace / DREAMPlace replace it with **eDensity**: treat macro area as electric
charge, solve the Poisson equation for the electrostatic potential, and use the
field energy as the penalty. Every charge repels every other charge through the
global potential.

The first port of this module rasterised macros as hard rectangles into bins.
That gave a *degenerate* gradient: a macro fully inside one bin contributes a
constant overlap regardless of where in the bin it sits, so the density map --
and therefore the electrostatic force -- had zero gradient in bin interiors.
Result: macros clumped, proxy +96%.

This version fixes that with **bilinear density splatting**: each macro is a
point charge of mass = macro area, splatted onto its four nearest bin centres
with bilinear interpolation weights. As a macro centre moves continuously, the
splat weights change continuously, so the density field -- and the
electrostatic gradient -- is nonzero everywhere. This is the continuous-density
model ePlace's eDensity actually requires.

Poisson solve (periodic boundary, plain ``torch.fft``):
1. Splat the bilinear density map ``rho`` (zero-meaned for charge neutrality).
2. ``rho_hat = rfft2(rho)``.
3. ``psi_hat = rho_hat / lambda`` where ``lambda`` are the discrete-Laplacian
   eigenvalues ``(2 - 2cos(2 pi u / rows)) + (2 - 2cos(2 pi v / cols))``.
4. ``psi = irfft2(psi_hat)``; penalty = electrostatic energy ``sum(rho * psi)``.

Everything else (smooth HPWL, hard-macro overlap barrier, push-apart
legaliser, shelf-pack fallback, deadline handling, CUDA determinism) is
inherited unchanged from :class:`AnalyticalPlacer`.
"""

import time

import torch

from .engine import (
    AnalyticalPlacer,
    _build_pin_index,
    clearance_microns,
)


class AnalyticalPlacerV3(AnalyticalPlacer):
    """eDensity variant of :class:`AnalyticalPlacer` with bilinear splatting.

    ``edensity_weight`` scales the electrostatic penalty. Its units differ
    from the v1 ``density_weight`` (the energy term is a different quantity),
    so it is tuned independently.
    """

    def __init__(
        self,
        n_iters=400,
        lr=0.002,
        gamma_start=0.015,
        gamma_end=0.002,
        wl_weight=1.0,
        edensity_weight=1.0,
        edensity_ramp=1.0,
        overlap_weight=6.0,
        legalize_gap=0.003,
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
            density_weight=0.0,  # v3 uses eDensity instead of bin-overflow
            overlap_weight=overlap_weight,
            legalize_gap=legalize_gap,
            seed=seed,
            device=device,
            verbose=verbose,
        )
        self.edensity_weight = edensity_weight
        # edensity_ramp: the eDensity penalty starts at
        # ``edensity_weight * edensity_ramp`` and rises geometrically to the
        # full ``edensity_weight`` over the iteration schedule. ePlace ramps
        # the density penalty so the optimiser minimises wirelength first and
        # is pushed apart gradually -- applying full weight from iter 0 fights
        # HPWL the whole way. ramp=1.0 disables it (constant full weight).
        self.edensity_ramp = edensity_ramp

    @staticmethod
    def _splat_density(pos, mass, rows, cols):
        """Bilinear-splat point charges of ``mass`` onto a [rows, cols] grid.

        ``pos`` is canvas-normalised macro centres in [0, 1]. Each macro splats
        its mass onto the four nearest bin centres with bilinear weights, so
        the resulting density map is a continuous, differentiable function of
        ``pos``. ``index_add`` keeps the scatter autograd-friendly: the
        gradient flows through the splat *weights* (the integer bin indices
        are constants, which is correct -- the position dependence lives in
        the weights, not the cell choice).
        """
        device = pos.device
        # Fractional grid position of each macro centre. Bin centre (i, j)
        # sits at ((j + 0.5)/cols, (i + 0.5)/rows), so gx = cx*cols - 0.5.
        gx = pos[:, 0] * cols - 0.5
        gy = pos[:, 1] * rows - 0.5
        j0 = torch.floor(gx).long()
        i0 = torch.floor(gy).long()
        wx = (gx - j0.float())
        wy = (gy - i0.float())
        j0c = j0.clamp(0, cols - 1)
        j1c = (j0 + 1).clamp(0, cols - 1)
        i0c = i0.clamp(0, rows - 1)
        i1c = (i0 + 1).clamp(0, rows - 1)

        grid = torch.zeros(rows * cols, device=device, dtype=pos.dtype)
        for ii, jj, ww in (
            (i0c, j0c, (1.0 - wx) * (1.0 - wy)),
            (i0c, j1c, wx * (1.0 - wy)),
            (i1c, j0c, (1.0 - wx) * wy),
            (i1c, j1c, wx * wy),
        ):
            idx = ii * cols + jj
            grid = grid.index_add(0, idx, mass * ww)
        return grid.reshape(rows, cols)

    @staticmethod
    def _laplacian_eigenvalues(rows, cols, device):
        """Discrete-Laplacian eigenvalue grid for the periodic Poisson solve.

        Shape ``[rows, cols//2 + 1]`` to align with ``torch.fft.rfft2``.
        """
        u = torch.arange(rows, device=device, dtype=torch.float32)
        v = torch.arange(cols // 2 + 1, device=device, dtype=torch.float32)
        wu = 2.0 - 2.0 * torch.cos(2.0 * torch.pi * u / rows)
        wv = 2.0 - 2.0 * torch.cos(2.0 * torch.pi * v / cols)
        lam = wu.unsqueeze(1) + wv.unsqueeze(0)
        lam[0, 0] = 1.0  # (0,0) mode carries no energy; avoid div-by-zero
        return lam

    @classmethod
    def _edensity_loss(cls, pos, mass, rows, cols, lam):
        """Electrostatic field energy of the bilinear-splatted macro density.

        ``mass`` is per-macro charge (macro area). Minimising the returned
        energy spreads the charge toward a uniform field.
        """
        density = cls._splat_density(pos, mass, rows, cols)
        rho = density - density.mean()  # charge neutrality for periodic solve
        rho_hat = torch.fft.rfft2(rho)
        psi_hat = rho_hat / lam
        psi = torch.fft.irfft2(psi_hat, s=density.shape)
        return (rho * psi).sum()

    def place(self, benchmark, deadline=None):
        """As :meth:`AnalyticalPlacer.place` but with the eDensity penalty."""
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
        # Per-macro charge = macro area in normalised units.
        mass = (sizes_n[:, 0] * sizes_n[:, 1]).detach()
        clr_um = clearance_microns(cw, ch)
        g_push = max(self.legalize_gap, clr_um / cw, clr_um / ch)
        lo = torch.minimum(half, torch.full_like(half, 0.5))
        hi = torch.maximum(1.0 - half, lo)
        fixed = benchmark.macro_fixed.to(device).bool()

        ref = benchmark.macro_positions.to(device).float()
        init = torch.clamp(ref / canvas, lo, hi)

        var = init.clone().detach().requires_grad_(True)
        opt = torch.optim.Adam([var], lr=self.lr)

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
        # eDensity needs at least a 2x2 grid for a meaningful FFT.
        lam = None
        if rows >= 2 and cols >= 2:
            lam = self._laplacian_eigenvalues(rows, cols, device)

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

            if lam is not None:
                edens = self._edensity_loss(var, mass, rows, cols, lam)
                # Geometric ramp of the eDensity penalty: start at
                # edensity_weight * edensity_ramp, reach edensity_weight at
                # the final iteration. ramp=1.0 -> constant full weight.
                ew = self.edensity_weight * (
                    self.edensity_ramp ** (1.0 - frac))
                loss = loss + ew * edens

            if tri is not None and ov_w > 0:
                ovl = self._overlap_loss(var, half, nh, tri)
                loss = loss + ov_w * ovl

            loss.backward()
            opt.step()

            with torch.no_grad():
                var.clamp_(lo, hi)
                var[fixed] = init[fixed]

            if self.verbose and (it % 100 == 0 or it == self.n_iters - 1):
                print(f"  [v3] iter {it:4d}  loss={loss.item():.5f}  "
                      f"gamma={gamma:.4f}")

        pos = var.detach().clone()
        pos[fixed] = init[fixed]

        pos, remaining = self._legalize_hard(
            pos, half, fixed, nh, lo, hi, g_push, deadline=deadline)
        if remaining > 0:
            if self.verbose:
                print(f"  [v3] {remaining} overlaps remain -> shelf-pack")
            shelf_gap = float(max(clr_um / cw, clr_um / ch))
            pos = self._shelf_pack(pos, sizes_n, fixed, nh, shelf_gap)

        out = pos * canvas
        out = torch.clamp(out, sizes / 2.0, canvas - sizes / 2.0)
        out[fixed] = ref[fixed]
        return out.cpu()
