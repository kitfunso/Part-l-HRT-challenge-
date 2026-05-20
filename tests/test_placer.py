"""Submission-readiness tests for hrt-placer.

These run against a locally-cloned challenge harness (skipped when absent).
The tests are intentionally small and deterministic; their job is to catch
legality / timeout / clearance / determinism regressions before submission,
not to score proxy cost.
"""

import os

import pytest
import torch

from hrt_placer.engine import AnalyticalPlacer
from hrt_placer.search import _is_legal

# Use a small IBM subset for speed; ibm01 is the smallest, ibm09 covers a
# medium-sized design. Tests that loop over benchmarks use IBM_SAMPLE; the
# full-fleet test uses ALL_IBMS.
IBM_SAMPLE = ["ibm01", "ibm09"]
ALL_IBMS = [f"ibm{i:02d}" for i in range(1, 19)
            if i not in (5,)]  # ibm05 is absent from the public ICCAD04 set
BASE = "external/MacroPlacement/Testcases/ICCAD04"


@pytest.fixture(scope="session")
def benchmark_ibm01(macro_place):
    bm, plc = macro_place.loader.load_benchmark_from_dir(f"{BASE}/ibm01")
    return bm, plc


def _has_cuda():
    return torch.cuda.is_available()


@pytest.mark.parametrize("name", IBM_SAMPLE)
def test_engine_returns_legal(macro_place, name):
    """The analytical engine must return a legal placement on each benchmark."""
    bm, _ = macro_place.loader.load_benchmark_from_dir(f"{BASE}/{name}")
    placer = AnalyticalPlacer(device="cuda" if _has_cuda() else "cpu")
    placement = placer.place(bm)
    assert _is_legal(placement, bm), f"engine produced illegal placement on {name}"


def test_placer_returns_legal(macro_place, benchmark_ibm01):
    """``MyPlacer().place(benchmark)`` must produce a legal placement.

    Uses a short HRT_TIME_BUDGET so the test runs in under a minute. The engine
    legaliser still fires under deadline pressure, so the result must be legal.
    """
    bm, _ = benchmark_ibm01
    os.environ["HRT_TIME_BUDGET"] = "60"
    try:
        # Import here so the env var is picked up.
        import importlib
        import placer
        importlib.reload(placer)
        p = placer.MyPlacer()
        placement = p.place(bm)
    finally:
        os.environ.pop("HRT_TIME_BUDGET", None)
    assert _is_legal(placement, bm), \
        "MyPlacer.place produced illegal placement under 60s budget"


@pytest.mark.skipif(not _has_cuda(),
                    reason="CUDA-determinism only meaningful on GPU")
def test_engine_determinism_cuda(macro_place, benchmark_ibm01):
    """Two engine runs with the same seed must produce reproducible placements.

    Guards the determinism pin (cudnn.deterministic=True, TF32 off, deterministic
    algorithms enabled) added to ``AnalyticalPlacer.place`` so the eval host
    and dev box do not drift on proxy cost. Sub-1e-4 micron drift in placement
    coordinates is acceptable; the proxy is computed against the bin grid and
    is insensitive to those last few float bits. Anything above 1e-4 indicates
    a real determinism break (cudnn benchmark mode, TF32 on, non-deterministic
    scatter without ``use_deterministic_algorithms``).
    """
    bm, _ = benchmark_ibm01
    p1 = AnalyticalPlacer(device="cuda", seed=0, n_iters=80).place(bm)
    p2 = AnalyticalPlacer(device="cuda", seed=0, n_iters=80).place(bm)
    assert torch.allclose(p1, p2, atol=1e-4, rtol=0.0), \
        "engine non-deterministic on CUDA with same seed; check determinism pin"


def test_clearance_microns_ng45_scale():
    """The clearance helper must return >=12 um on NG45-scale canvases.

    PRD requires this so Tier-2 auto-spacing does not silently override the
    submitted coordinates. NG45 canvases run 900-2100 um; the IBM dies are
    abstract units near 23 and intentionally fall to a proportional clearance
    via the ``< 200`` branch.
    """
    from hrt_placer.engine import clearance_microns
    assert clearance_microns(900.0, 900.0) >= 12.0 - 1e-9
    assert clearance_microns(2100.0, 2100.0) >= 12.0 - 1e-9
    # IBM-scale (abstract units): proportional, not clamped at 12 um.
    clr_ibm = clearance_microns(23.0, 23.0)
    assert clr_ibm < 12.0, "IBM-scale die should use proportional clearance"
    assert clr_ibm > 0.0


def test_timeout_returns_legal(macro_place, benchmark_ibm01):
    """An aggressively short engine deadline still returns a legal placement.

    Hits the shelf-pack legaliser fallback path. If this regresses, the
    submission risks returning an illegal placement on slow eval hosts.
    """
    import time
    bm, _ = benchmark_ibm01
    # Force the deadline to fire almost immediately. The shelf-pack fallback
    # must still produce a legal placement.
    deadline = time.time() + 1.0
    placer = AnalyticalPlacer(device="cuda" if _has_cuda() else "cpu",
                              n_iters=50)
    placement = placer.place(bm, deadline=deadline)
    assert _is_legal(placement, bm), \
        "engine with immediate deadline did not legalise via shelf-pack"


def test_portfolio_logs_winner(macro_place, benchmark_ibm01, capfd):
    """``MyPlacer.place`` must log the portfolio winner to stderr.

    Guards the visible-failure contract added to placer.py: if the portfolio
    silently regresses, the eval host log is the only way to notice.
    """
    bm, _ = benchmark_ibm01
    os.environ["HRT_TIME_BUDGET"] = "60"
    try:
        import importlib
        import placer
        importlib.reload(placer)
        p = placer.MyPlacer()
        p.place(bm)
    finally:
        os.environ.pop("HRT_TIME_BUDGET", None)
    err = capfd.readouterr().err
    assert "portfolio winner" in err, \
        "MyPlacer.place did not log a portfolio winner line to stderr"
