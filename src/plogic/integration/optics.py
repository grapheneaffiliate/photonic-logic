from __future__ import annotations

from typing import Optional

import numpy as np


def epsilon_r_nonlinear(
    omega: float,
    T_op: float,
    P_ctrl: float,
    *,
    n2: float | None = None,
    gXPM: float | None = None,
    A_eff: float = 0.6e-12,
    n_eff: float = 3.4,
    g_geom: float = 1.0,
) -> float:
    """
    Compute effective relative permittivity epsilon_r,eff under XPM control drive.

    Parameters
    ----------
    omega : float
        Optical angular frequency (rad/s). Currently only used for signature completeness.
    T_op : float
        Operating temperature in K (placeholder for P2; no direct effect here).
    P_ctrl : float
        Control optical power in Watts.
    n2 : float | None
        Kerr nonlinear index (m^2/W). If provided, used as primary path for Δn.
    gXPM : float | None
        Calibrated shortcut gain relating P_ctrl to Δn (1/W equivalent in practice),
        used only if n2 is None.
    A_eff : float
        Effective mode area (m^2).
    n_eff : float
        Effective linear refractive index (unitless).
    g_geom : float
        Geometry factor mapping Δn to effective detuning/phase pull scaling (unitless).

    Returns
    -------
    float
        Effective relative permittivity epsilon_r,eff ≈ (n_eff + Δn_eff)^2.
    """
    # Compute Δn from either Kerr n2 (preferred) or calibrated gXPM
    if n2 is not None:
        I = P_ctrl / max(A_eff, 1e-24)  # W / m^2
        delta_n = n2 * I
    elif gXPM is not None:
        # Map power to an effective index shift via calibration constant (dimension proxy).
        # The exact physical meaning depends on calibration; this path is a fallback.
        delta_n = gXPM * P_ctrl
    else:
        delta_n = 0.0

    # Map to an effective Δn including geometry factor (dimensionless scaling)
    delta_n_eff = g_geom * delta_n

    # Effective permittivity approximation
    n_total = n_eff + delta_n_eff
    epsilon_r_eff = float(n_total * n_total)
    return epsilon_r_eff


def delta_omega_xpm(
    omega0: float,
    T_op: float,
    P_ctrl: float,
    *,
    n2: float | None = None,
    gXPM: float | None = None,
    A_eff: float = 0.6e-12,
    n_eff: float = 3.4,
    g_geom: float = 1.0,
) -> float:
    """
    Compute cavity detuning Δω induced by XPM at control power P_ctrl.

    Model:
        Δn = n2 * (P_ctrl / A_eff)             (preferred physics path)
        or Δn ≈ gXPM * P_ctrl                  (calibrated shortcut if no n2)

        Δω ≈ -(ω0 / n_eff) * g_geom * Δn

    Notes
    -----
    - Sign convention: For positive Δn and positive g_geom, Δω is negative
      (red-shift). Adjust g_geom if your geometry induces different sign.
    - T_op is accepted for P2 compatibility (e.g., temperature-dependent parameters).
    """
    if n2 is not None:
        I = P_ctrl / max(A_eff, 1e-24)  # W / m^2
        delta_n = n2 * I
    elif gXPM is not None:
        delta_n = gXPM * P_ctrl
    else:
        delta_n = 0.0

    # Convert index change to frequency pull
    delta_omega = -(omega0 / max(n_eff, 1e-9)) * g_geom * delta_n
    return float(delta_omega)
