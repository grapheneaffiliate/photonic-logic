import math
import pytest
from plogic.controller import PhotonicMolecule, TWOPI


@pytest.fixture
def device() -> PhotonicMolecule:
    """Standard test device with canonical parameters."""
    GHz = TWOPI * 1e9
    return PhotonicMolecule(
        kappa_A=0.39*GHz, 
        kappa_B=0.39*GHz,
        kappa_eA=0.20*GHz, 
        kappa_eB=0.20*GHz,
        J=1.5*GHz,
        # keep other defaults
    )
