from plogic.controller import PhotonicMolecule


def test_basic_response_bounds():
    dev = PhotonicMolecule()
    r = dev.steady_state_response(dev.omega0, 0.0)
    assert 0.0 <= r["T_through"] <= 1.0
    assert 0.0 <= r["T_drop"] <= 1.0


def test_device_creation():
    """Test that device can be created with default parameters."""
    dev = PhotonicMolecule()
    assert dev.omega0 > 0
    assert dev.kappa_A > 0
    assert dev.J > 0
