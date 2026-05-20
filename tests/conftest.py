"""pytest setup for hrt-placer tests.

Adds the repo's ``src`` to ``sys.path`` and, when present, chdirs into the
locally-cloned challenge harness so ``macro_place.loader`` etc. resolve. The
harness is not checked in (judges drop their own around the submission); tests
that need it skip gracefully via the ``challenge`` fixture below.
"""

import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHALLENGE = os.path.join(REPO, "external", "macro-place-challenge-2026")
SRC = os.path.join(REPO, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
if REPO not in sys.path:
    sys.path.insert(0, REPO)


@pytest.fixture(scope="session")
def challenge():
    """Return the path to the local challenge harness, or skip if absent."""
    if not os.path.isdir(CHALLENGE):
        pytest.skip(f"challenge harness not found at {CHALLENGE} "
                    "(local-dev clone only; see README)")
    if CHALLENGE not in sys.path:
        sys.path.insert(0, CHALLENGE)
    # chdir so macro_place.loader's relative paths resolve
    os.chdir(CHALLENGE)
    return CHALLENGE


@pytest.fixture(scope="session")
def macro_place(challenge):
    """Import the challenge harness modules; skip if not importable."""
    try:
        from macro_place import loader, objective, utils
    except ImportError as exc:
        pytest.skip(f"macro_place harness not importable: {exc}")
    return type("MP", (), {"loader": loader, "objective": objective,
                           "utils": utils})
