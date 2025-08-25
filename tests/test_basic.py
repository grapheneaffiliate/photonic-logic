import numpy as np
from plogic.controller import PhotonicMolecule, TWOPI

def test_basic_response_bounds():
    dev = PhotonicMolecule()
    r = dev.steady_state_response(dev.omega0, 0.0)
    assert 0.0 <= r['T_through'] <= 1.0
    assert 0.0 <= r['T_drop'] <= 1.0

def test_monotone_trend():
    dev = PhotonicMolecule()
    P = np.linspace(0, 2e-3, 11)
    T = [dev.steady_state_response(dev.omega0, p)['T_through'] for p in P]
    # should not be strictly constant
    assert not all(abs(T[i+1]-T[i]) < 1e-12 for i in range(len(T)-1))
