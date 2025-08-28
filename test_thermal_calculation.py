#!/usr/bin/env python
"""Test the actual thermal calculation to find the bug."""

import sys
import os
import math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from plogic.analysis import PowerInputs, compute_power_report
from plogic.materials import PlatformDB

def test_thermal_calculation():
    """Test the thermal calculation with explicit values."""
    
    print("Testing thermal calculation with explicit values...")
    print("=" * 60)
    
    # Load AlGaAs platform
    pdb = PlatformDB()
    platform = pdb.get("AlGaAs")
    
    # Test parameters
    P_high_mW = 0.06
    pulse_ns = 1.4
    wavelength_nm = 1550
    link_length_um = 60.0
    
    # Manual calculation
    print("Manual Calculation:")
    print("-" * 40)
    
    # Convert units
    P_high_W = P_high_mW * 1e-3
    t_switch_ns = pulse_ns
    t_s = t_switch_ns * 1e-9
    
    # Material properties
    n2 = platform.nonlinear.n2_m2_per_W
    Aeff_um2 = platform.nonlinear.Aeff_um2_default
    Aeff_m2 = Aeff_um2 * 1e-12
    dn_dT = platform.thermal.dn_dT_per_K
    tau_thermal_ns = platform.thermal.tau_thermal_ns
    tau_th_s = tau_thermal_ns * 1e-9
    loss_dB_cm = platform.fabrication.loss_dB_per_cm
    
    print(f"P_high: {P_high_mW} mW = {P_high_W} W")
    print(f"t_switch: {t_switch_ns} ns = {t_s} s")
    print(f"tau_thermal: {tau_thermal_ns} ns = {tau_th_s} s")
    print(f"n2: {n2} m²/W")
    print(f"Aeff: {Aeff_um2} µm² = {Aeff_m2} m²")
    print(f"dn/dT: {dn_dT} /K")
    print(f"Loss: {loss_dB_cm} dB/cm")
    print()
    
    # Calculate intensity and Kerr
    I_W_m2 = P_high_W / Aeff_m2
    delta_n_kerr = n2 * I_W_m2
    print(f"Intensity: {I_W_m2:.2e} W/m²")
    print(f"Δn_Kerr: {delta_n_kerr:.2e}")
    print()
    
    # Calculate absorption
    alpha_m = loss_dB_cm * 100.0 * math.log(10.0) / 10.0
    L_eff_um = 10.0  # Default from PowerInputs
    L_eff_m = L_eff_um * 1e-6
    P_abs_W = P_high_W * (1.0 - math.exp(-alpha_m * L_eff_m))
    
    print(f"alpha: {alpha_m:.4f} /m")
    print(f"L_eff: {L_eff_um} µm = {L_eff_m} m")
    print(f"P_abs: {P_abs_W:.2e} W")
    print()
    
    # Calculate drift - THE KEY CALCULATION
    absorption_ratio = P_abs_W / P_high_W
    time_ratio = t_s / tau_th_s
    drift = absorption_ratio * time_ratio
    
    print(f"P_abs/P_high: {absorption_ratio:.6e}")
    print(f"t_s/tau_th_s: {time_ratio:.6e}")
    print(f"drift: {drift:.6e}")
    print()
    
    # Calculate thermal index change
    k_th = 1.0  # Default thermal scale
    delta_n_thermal = k_th * dn_dT * drift
    thermal_ratio = delta_n_thermal / delta_n_kerr
    
    print(f"k_th: {k_th}")
    print(f"Δn_thermal: {delta_n_thermal:.2e}")
    print(f"Thermal ratio: {thermal_ratio:.4f}")
    print()
    
    # Now run the actual function
    print("=" * 60)
    print("Actual PowerReport Calculation:")
    print("-" * 40)
    
    pins = PowerInputs(
        wavelength_nm=wavelength_nm,
        platform_loss_dB_cm=loss_dB_cm,
        coupling_eta=0.98,
        link_length_um=link_length_um,
        fanout=1,
        pulse_ns=pulse_ns,
        P_high_mW=P_high_mW,
        threshold_norm=0.5,
        worst_off_norm=0.01,
        extinction_target_dB=21.0,
        n2_m2_per_W=n2,
        Aeff_um2=Aeff_um2,
        dn_dT_per_K=dn_dT,
        tau_thermal_ns=tau_thermal_ns,
        thermal_scale=1.0,
        L_eff_um=10.0,
        include_2pa=platform.flags.tpa_present_at_1550,
        beta_2pa_m_per_W=platform.nonlinear.beta_2pa_m_per_W,
        auto_timing=False
    )
    
    report = compute_power_report(pins)
    
    thermal_info = report.raw.get("thermal", {})
    thermal_raw = report.raw.get("thermal_raw", {})
    
    print(f"Δn_Kerr: {thermal_info.get('delta_n_kerr', 0):.2e}")
    print(f"Δn_thermal: {thermal_info.get('delta_n_thermal', 0):.2e}")
    print(f"Thermal ratio: {thermal_info.get('thermal_ratio', 0):.4f}")
    print(f"Thermal flag: {thermal_info.get('thermal_flag', 'unknown')}")
    
    if thermal_raw:
        print()
        print("Raw debug values:")
        print(f"  I_W_per_m2: {thermal_raw.get('I_W_per_m2', 0):.2e}")
        print(f"  P_abs_W: {thermal_raw.get('P_abs_W', 0):.2e}")
        print(f"  tau_th_s: {thermal_raw.get('tau_th_s', 0):.2e}")
        print(f"  drift_unscaled: {thermal_raw.get('drift_unscaled', 0):.2e}")
        print(f"  drift_clamped: {thermal_raw.get('drift_clamped', 0):.2e}")
        print(f"  thermal_scale: {thermal_raw.get('thermal_scale', 0)}")
    
    print()
    print("=" * 60)
    print("COMPARISON:")
    
    expected_ratio = thermal_ratio
    actual_ratio = thermal_info.get('thermal_ratio', 0)
    
    print(f"Expected thermal ratio: {expected_ratio:.4f}")
    print(f"Actual thermal ratio: {actual_ratio:.4f}")
    
    if abs(actual_ratio - expected_ratio) < 0.01:
        print("✓ Thermal calculation matches!")
    else:
        print("✗ Thermal calculation MISMATCH!")
        print(f"  Error factor: {actual_ratio / expected_ratio:.2e}")
        
        # Check if the error is a power of 10
        error_factor = actual_ratio / expected_ratio
        log_error = math.log10(abs(error_factor))
        if abs(log_error - round(log_error)) < 0.1:
            print(f"  This looks like a 10^{round(log_error)} unit conversion error!")

if __name__ == "__main__":
    test_thermal_calculation()
