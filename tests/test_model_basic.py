import math
import numpy as np
import pytest

from plogic.controller import PhotonicMolecule, TWOPI


def test_passivity_and_bounds():
    dev = PhotonicMolecule()
    omega = dev.omega0
    # Sweep a small neighborhood around resonance and a small control sweep
    omegas = np.linspace(omega - 3 * dev.J, omega + 3 * dev.J, 21)
    powers = np.linspace(0.0, 2e-3, 5)
    for w in omegas:
        for P in powers:
            r = dev.steady_state_response(w, P)
            Tt = r["T_through"]
            Td = r["T_drop"]
            assert 0.0 <= Tt <= 1.0
            assert 0.0 <= Td <= 1.0
            assert (Tt + Td) <= 1.0001  # allow tiny slack for FP rounding


def test_xpm_trend_not_constant():
    dev = PhotonicMolecule()
    P = np.linspace(0.0, 2e-3, 9)
    T = [dev.steady_state_response(dev.omega0, p)["T_through"] for p in P]
    # Should not be strictly constant across the control sweep
    diffs = np.diff(T)
    assert np.any(np.abs(diffs) > 1e-9)
