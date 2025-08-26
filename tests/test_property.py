from hypothesis import given
from hypothesis import strategies as st

from plogic.controller import PhotonicMolecule


@given(st.floats(min_value=0.0, max_value=5e-3))
def test_power_passivity(P_ctrl: float) -> None:
    """
    Property-based test: passivity/energy bound at the through/drop/reflect ports.
    For physically reasonable parameters, the sum of powers should not exceed 1.
    """
    dev = PhotonicMolecule()
    omega = dev.omega0
    resp = dev.steady_state_response(omega, P_ctrl=P_ctrl)
    t = float(resp["T_through"])
    d = float(resp["T_drop"])
    r = float(resp["R_reflect"])
    # Allow a small slack due to simplified scattering model and numerical clipping.
    assert t + d + r <= 1.0 + 0.05
