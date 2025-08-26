import json

import numpy as np

from plogic.controller import PhotonicMolecule, TWOPI
from plogic.integration import delta_omega_xpm


def test_delta_omega_scaling_and_sign():
    omega0 = TWOPI * 193.5e12
    n2 = 1e-17  # Kerr coefficient (m^2/W), chosen to yield a noticeable but small shift
    A_eff = 0.6e-12
    n_eff = 3.4
    g_geom = 1.0

    d0 = delta_omega_xpm(omega0, T_op=300.0, P_ctrl=0.0, n2=n2, A_eff=A_eff, n_eff=n_eff, g_geom=g_geom)
    d1 = delta_omega_xpm(omega0, T_op=300.0, P_ctrl=1e-3, n2=n2, A_eff=A_eff, n_eff=n_eff, g_geom=g_geom)
    d2 = delta_omega_xpm(omega0, T_op=300.0, P_ctrl=2e-3, n2=n2, A_eff=A_eff, n_eff=n_eff, g_geom=g_geom)

    # At zero power, no detuning
    assert d0 == 0.0
    # Positive n2 and positive g_geom produce a red-shift (negative Δω), increasing in magnitude with P_ctrl
    assert d1 < 0.0
    assert abs(d2) > abs(d1)


def test_response_monotonic_trend_with_power_small_range():
    # Physics-driven XPM mode should produce a monotonic trend in transmission near resonance
    n2 = 1e-17
    dev = PhotonicMolecule(xpm_mode="physics", n2=n2, A_eff=0.6e-12, n_eff=3.4, g_geom=1.0)
    omega = dev.omega0

    # Use a small set of powers to check monotonic behavior
    powers = [0.0, 5e-3, 10e-3]
    T = [dev.steady_state_response(omega, P)["T_through"] for P in powers]

    # Differences should have the same sign (monotonic non-decreasing or non-increasing)
    d1 = T[1] - T[0]
    d2 = T[2] - T[1]
    assert d1 * d2 >= 0.0
