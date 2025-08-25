from dataclasses import asdict

import numpy as np

from plogic.controller import PhotonicMolecule


def test_dataclass_roundtrip():
    dev = PhotonicMolecule()
    data = asdict(dev)
    dev2 = PhotonicMolecule(**data)
    assert dev2 == dev


def test_eigenfrequencies_types_and_ordering():
    dev = PhotonicMolecule()
    w_plus, w_minus = dev.eigenfrequencies()
    # Types
    assert isinstance(w_plus, complex)
    assert isinstance(w_minus, complex)
    # Real parts should be symmetric around 0 when kappa_A == kappa_B
    reals = np.sort([w_plus.real, w_minus.real])
    # Expect approximately [-J, +J]
    assert np.isclose(reals[0], -dev.J, rtol=0.15, atol=0.0)
    assert np.isclose(reals[1], dev.J, rtol=0.15, atol=0.0)
