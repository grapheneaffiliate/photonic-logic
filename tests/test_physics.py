import numpy as np

from plogic.controller import PhotonicMolecule


def _sweep_transmission(dev: PhotonicMolecule, omegas: np.ndarray, P: float = 0.0):
    Tt = []
    Td = []
    for w in omegas:
        r = dev.steady_state_response(w, P)
        Tt.append(r["T_through"])
        Td.append(r["T_drop"])
    return np.array(Tt), np.array(Td)


def test_passivity_bounds():
    dev = PhotonicMolecule()
    omegas = np.linspace(dev.omega0 - 5 * dev.J, dev.omega0 + 5 * dev.J, 201)
    Tt, Td = _sweep_transmission(dev, omegas, P=0.5e-3)
    assert np.all(Tt >= 0) and np.all(Tt <= 1)
    assert np.all(Td >= 0) and np.all(Td <= 1)
    # Through + drop should not exceed 1 (allow tiny FP slack)
    assert np.all(Tt + Td <= 1.0001)


def test_resolved_splitting_minima_near_plus_minus_J():
    dev = PhotonicMolecule()
    J = dev.J
    # Sweep around resonance; minima in T_through should occur near omega0 ± J
    omegas = np.linspace(dev.omega0 - 5 * J, dev.omega0 + 5 * J, 2001)
    Tt, _ = _sweep_transmission(dev, omegas, P=0.0)
    # Find two smallest values (minima) of T_through
    idx = np.argpartition(Tt, 2)[:2]
    w_min = np.sort(omegas[idx])
    # Expect close to omega0 - J and omega0 + J within a reasonable tolerance
    assert np.isclose(w_min[0], dev.omega0 - J, rtol=0.2, atol=0.0)
    assert np.isclose(w_min[1], dev.omega0 + J, rtol=0.2, atol=0.0)
