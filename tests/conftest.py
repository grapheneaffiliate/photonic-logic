import os
import sys

# Ensure local 'src' takes precedence over any installed distribution
_THIS_DIR = os.path.dirname(__file__)
_SRC_PATH = os.path.abspath(os.path.join(_THIS_DIR, "..", "src"))
if _SRC_PATH not in sys.path:
    sys.path.insert(0, _SRC_PATH)

import pytest

from plogic.controller import TWOPI, PhotonicMolecule


@pytest.fixture
def device() -> PhotonicMolecule:
    """Standard test device with canonical parameters."""
    GHz = TWOPI * 1e9
    return PhotonicMolecule(
        kappa_A=0.39 * GHz,
        kappa_B=0.39 * GHz,
        kappa_eA=0.20 * GHz,
        kappa_eB=0.20 * GHz,
        J=1.5 * GHz,
        # keep other defaults
    )
