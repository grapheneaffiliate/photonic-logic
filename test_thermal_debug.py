#!/usr/bin/env python
"""Debug thermal calculation to understand why ratio is so high."""

import sys
import os
import math

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from plogic.analysis import PowerInputs, compute_power_report
from plogic.analysis.power import loss_dBcm_to_alpha_m
from plogic.materials import PlatformDB

def debug_thermal():
    """Debug thermal calculation step by step."""
    
    print("Debugging Thermal Calculation")
    print("=" * 50)
    
    # Load AlGaAs platform
    pdb = PlatformDB()
    platform = pdb.get("AlGaAs")
    
    # Parameters
    P_high_mW = 0.06
    pulse_ns = 1.4
    wavelength_nm = platform.default_wavelength_nm
    Aeff_um2 = platform.nonlinear.Aeff_um2_default
    n2 = platform.nonlinear.n2_m2_per_W
    dn_dT = platform.thermal.dn_dT_per_K
    tau_thermal_ns = platform.thermal.tau_thermal_ns
    loss_dB_cm = platform.fabrication.loss_dB_per_cm
    L_eff_um = 10.0
    
    print(f"Input Parameters:")
    print(f"  P_high: {P_high_mW} mW = {P_high_mW*1e-3} W")
    print(f"  Pulse: {pulse_ns} ns")
    print(f"  Wavelength: {wavelength_nm} nm")
    print(f"  A_eff: {Aeff_um2} µm² = {Aeff_um2*1e-12} m²")
    print(f"  n2: {n2} m²/W")
    print(f"  dn/dT: {dn_dT} /K")
    print(f"  τ_thermal: {tau_thermal_ns} ns = {tau_thermal_ns*1e-9} s")
    print(f"  Loss: {loss_dB_cm} dB/cm")
    print(f"  L_eff: {L_eff_um} µm = {L_eff_um*1e-6} m")
    print()
    
    # Manual calculation
    P_high_W = P_high_mW * 1e-3
    Aeff_m2 = Aeff_um2 * 1e-12
    L_eff_m = L_eff_um * 1e-6
    t_switch_s = pulse_ns * 1e-9
    tau_th_s = tau_thermal_ns * 1e-9
    
    # Intensity
    I_W_m2 = P_high_W / Aeff_m2
    print(f"Intensity Calculation:")
    print(f"  I = P/A = {P_high_W:.6f} W / {Aeff_m2:.2e} m²")
    print(f"  I = {I_W_m2:.2e} W/m²")
    print()
    
    # Kerr index change
    delta_n_kerr = n2 * I_W_m2
    print(f"Kerr Index Change:")
    print(f"  Δn_Kerr = n2 × I = {n2:.2e} × {I_W_m2:.2e}")
    print(f"  Δn_Kerr = {delta_n_kerr:.2e}")
    print()
    
    # Absorbed power
    alpha_m = loss_dBcm_to_alpha_m(loss_dB_cm)
    P_abs_W = P_high_W * (1.0 - math.exp(-alpha_m * L_eff_m))
    print(f"Absorbed Power:")
    print(f"  α = {alpha_m:.4f} /m")
    print(f"  P_abs = P × (1 - exp(-α×L))")
    print(f"  P_abs = {P_high_W:.6f} × (1 - exp(-{alpha_m:.4f}×{L_eff_m:.2e}))")
    print(f"  P_abs = {P_abs_W:.2e} W")
    print()
    
    # Thermal drift
    drift_raw = (P_abs_W / P_high_W) * (t_switch_s / tau_th_s)
    drift = min(max(drift_raw, 0.0), 10.0)
    print(f"Thermal Drift:")
    print(f"  drift = (P_abs/P) × (t_switch/τ_thermal)")
    print(f"  drift = ({P_abs_W:.2e}/{P_high_W:.6f}) × ({t_switch_s:.2e}/{tau_th_s:.2e})")
    print(f"  drift = {P_abs_W/P_high_W:.6f} × {t_switch_s/tau_th_s:.2e}")
    print(f"  drift_raw = {drift_raw:.2e}")
    print(f"  drift_clamped = {drift:.2e}")
    print()
    
    # Thermal index change (with k_th = 1.0)
    k_th = 1.0
    delta_n_thermal = k_th * dn_dT * drift
    print(f"Thermal Index Change:")
    print(f"  k_th = {k_th}")
    print(f"  Δn_thermal = k_th × dn/dT × drift")
    print(f"  Δn_thermal = {k_th} × {dn_dT:.2e} × {drift:.2e}")
    print(f"  Δn_thermal = {delta_n_thermal:.2e}")
    print()
    
    # Thermal ratio
    thermal_ratio = delta_n_thermal / delta_n_kerr
    print(f"Thermal Ratio:")
    print(f"  ratio = Δn_thermal / Δn_Kerr")
    print(f"  ratio = {delta_n_thermal:.2e} / {delta_n_kerr:.2e}")
    print(f"  ratio = {thermal_ratio:.6f}")
    print()
    
    # Platform thresholds for AlGaAs
    print(f"AlGaAs Thermal Thresholds:")
    print(f"  OK: < 0.5")
    print(f"  Caution: 0.5 - 2.0")
    print(f"  Danger: > 2.0")
    print()
    
    if thermal_ratio < 0.5:
        status = "OK"
    elif thermal_ratio < 2.0:
        status = "CAUTION"
    else:
        status = "DANGER"
    
    print(f"Thermal Status: {status}")
    print()
    
    # Now run the actual function to compare
    pins = PowerInputs(
        wavelength_nm=wavelength_nm,
        platform_loss_dB_cm=loss_dB_cm,
        coupling_eta=0.98,
        link_length_um=60.0,
        fanout=1,
        pulse_ns=pulse_ns,
        P_high_mW=P_high_mW,
        threshold_norm=0.5,
        worst_off_norm=0.01,
        extinction_target_dB=21.0,
        er_epsilon=1e-12,
        n2_m2_per_W=n2,
        Aeff_um2=Aeff_um2,
        dn_dT_per_K=dn_dT,
        tau_thermal_ns=tau_thermal_ns,
        L_eff_um=L_eff_um,
        include_2pa=False,
        beta_2pa_m_per_W=0.0,
        auto_timing=False
    )
    
    power_rep = compute_power_report(pins)
    thermal_info = power_rep.raw.get("thermal", {})
    thermal_raw = power_rep.raw.get("thermal_raw", {})
    
    print("=" * 50)
    print("Actual Function Results:")
    print(f"  Δn_Kerr: {thermal_info.get('delta_n_kerr', 0):.2e}")
    print(f"  Δn_thermal: {thermal_info.get('delta_n_thermal', 0):.2e}")
    print(f"  Thermal ratio: {thermal_info.get('thermal_ratio', 0):.6f}")
    print(f"  Thermal flag: {thermal_info.get('thermal_flag', 'unknown')}")
    
    if thermal_raw:
        print("\nRaw Debug Values:")
        print(f"  I_W_per_m2: {thermal_raw.get('I_W_per_m2', 0):.2e}")
        print(f"  P_abs_W: {thermal_raw.get('P_abs_W', 0):.2e}")
        print(f"  drift_unscaled: {thermal_raw.get('drift_unscaled', 0):.2e}")
        print(f"  drift_clamped: {thermal_raw.get('drift_clamped', 0):.2e}")
        print(f"  thermal_scale: {thermal_raw.get('thermal_scale', 0)}")

if __name__ == "__main__":
    debug_thermal()
