import numpy as np

from plogic.controller import PhotonicMolecule


def test_basic_response_bounds():
    dev = PhotonicMolecule()
    r = dev.steady_state_response(dev.omega0, 0.0)
    assert 0.0 <= r["T_through"] <= 1.0
    assert 0.0 <= r["T_drop"] <= 1.0


def test_monotone_trend():
    dev = PhotonicMolecule()
    # At a fixed frequency, the phase should vary with control power due to XPM-induced detuning.
    P = np.linspace(0, 2e-3, 11)
    phases = [dev.steady_state_response(dev.omega0, p)["phase_through"] for p in P]
    diffs = np.diff(phases)
    assert np.any(np.abs(diffs) > 1e-12)
