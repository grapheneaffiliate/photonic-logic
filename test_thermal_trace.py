#!/usr/bin/env python
"""Trace thermal calculation to find unit conversion issue."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from plogic.analysis import PowerInputs, compute_power_report
from plogic.materials import PlatformDB
import math

def trace_thermal_calculation():
    """Trace the thermal calculation step by step."""
    
    print("Tracing thermal calculation for AlGaAs...")
    print("=" * 60)
    
    # Load AlGaAs platform
    pdb = PlatformDB()
    platform = pdb.get("AlGaAs")
    
    # Test parameters (same as cascade test)
    P_high_mW = 0.06
    pulse_ns = 1.4
    wavelength_nm = 1550
    link_length_um = 60.0
    
    # Manual calculation to trace the issue
    print("\n1. Input Parameters:")
    print(f"   P_high: {P_high_mW} mW = {P_high_mW*1e-3} W")
    print(f"   Pulse: {pulse_ns} ns = {pulse_ns*1e-9} s")
    print(f"   Link length: {link_length_um} µm = {link_length_um*1e-6} m")
    
    print("\n2. Platform Properties:")
    print(f"   n2: {platform.nonlinear.n2_m2_per_W} m²/W")
    print(f"   Aeff: {platform.nonlinear.Aeff_um2_default} µm² = {platform.nonlinear.Aeff_um2_default*1e-12} m²")
    print(f"   dn/dT: {platform.thermal.dn_dT_per_K} /K")
    print(f"   τ_thermal: {platform.thermal.tau_thermal_ns} ns")
    print(f"   thermal_scale: {platform.thermal.thermal_scale}")
    print(f"   Loss: {platform.fabrication.loss_dB_per_cm} dB/cm")
    
    print("\n3. Thermal Calculation Steps:")
    
    # Convert units
    P_high_W = P_high_mW * 1e-3
    t_switch_ns = pulse_ns
    t_s = t_switch_ns * 1e-9
    Aeff_m2 = platform.nonlinear.Aeff_um2_default * 1e-12
    
    print(f"   P_high_W: {P_high_W}")
    print(f"   t_switch_ns: {t_switch_ns}")
    print(f"   t_s (seconds): {t_s}")
    print(f"   Aeff_m2: {Aeff_m2}")
    
    # Intensity
    I_W_m2 = P_high_W / Aeff_m2
    print(f"\n   Intensity I_W_m2: {I_W_m2:.2e}")
    
    # Kerr effect
    delta_n_kerr = platform.nonlinear.n2_m2_per_W * I_W_m2
    print(f"   Δn_Kerr: {delta_n_kerr:.2e}")
    
    # Absorption calculation
    alpha_m = platform.fabrication.loss_dB_per_cm * 100.0 * math.log(10.0) / 10.0
    L_eff_m = 10.0 * 1e-6  # Default L_eff from power.py
    P_abs_W = P_high_W * (1.0 - math.exp(-alpha_m * L_eff_m))
    
    print(f"\n   alpha_m: {alpha_m:.4f}")
    print(f"   L_eff_m: {L_eff_m}")
    print(f"   P_abs_W: {P_abs_W:.2e}")
    
    # Thermal time constant - THIS IS THE KEY PART
    tau_thermal_ns = platform.thermal.tau_thermal_ns
    print(f"\n   τ_thermal from database: {tau_thermal_ns} ns")
    
    # Check different interpretations
    tau_th_s_correct = tau_thermal_ns * 1e-9  # Convert ns to s
    tau_th_s_wrong = tau_thermal_ns  # If treated as seconds by mistake
    
    print(f"   τ_thermal as seconds (correct): {tau_th_s_correct:.2e} s")
    print(f"   τ_thermal if wrongly used as-is: {tau_th_s_wrong} s")
    
    # Calculate drift both ways
    print(f"\n4. Drift Calculation:")
    print(f"   Formula: drift = (P_abs/P_high) * (t_switch/τ_thermal)")
    
    drift_correct = (P_abs_W / P_high_W) * (t_s / tau_th_s_correct)
    drift_wrong = (P_abs_W / P_high_W) * (t_s / tau_th_s_wrong)
    
    print(f"   With correct τ_thermal ({tau_th_s_correct:.2e} s):")
    print(f"      drift = {drift_correct:.6f}")
    
    print(f"   With wrong τ_thermal ({tau_th_s_wrong} s):")
    print(f"      drift = {drift_wrong:.2e}")
    
    # Thermal index change
    k_th = platform.thermal.thermal_scale
    dn_dT = platform.thermal.dn_dT_per_K
    
    delta_n_thermal_correct = k_th * dn_dT * drift_correct
    delta_n_thermal_wrong = k_th * dn_dT * drift_wrong
    
    print(f"\n5. Thermal Index Change:")
    print(f"   k_th (thermal_scale): {k_th}")
    print(f"   dn/dT: {dn_dT}")
    
    print(f"\n   With correct units:")
    print(f"      Δn_thermal = {delta_n_thermal_correct:.2e}")
    print(f"      Ratio = {delta_n_thermal_correct/delta_n_kerr:.4f}")
    
    print(f"\n   With wrong units:")
    print(f"      Δn_thermal = {delta_n_thermal_wrong:.2e}")
    print(f"      Ratio = {delta_n_thermal_wrong/delta_n_kerr:.2f}")
    
    # Now run the actual computation to see what it gives
    print("\n" + "=" * 60)
    print("6. Actual PowerReport Calculation:")
    
    pins = PowerInputs(
        wavelength_nm=wavelength_nm,
        platform_loss_dB_cm=platform.fabrication.loss_dB_per_cm,
        coupling_eta=0.98,
        link_length_um=link_length_um,
        fanout=1,
        pulse_ns=pulse_ns,
        P_high_mW=P_high_mW,
        threshold_norm=0.5,
        worst_off_norm=0.01,
        extinction_target_dB=21.0,
        n2_m2_per_W=platform.nonlinear.n2_m2_per_W,
        Aeff_um2=platform.nonlinear.Aeff_um2_default,
        dn_dT_per_K=platform.thermal.dn_dT_per_K,
        tau_thermal_ns=platform.thermal.tau_thermal_ns,
        thermal_scale=platform.thermal.thermal_scale,
        include_2pa=platform.flags.tpa_present_at_1550,
        beta_2pa_m_per_W=platform.nonlinear.beta_2pa_m_per_W,
        auto_timing=False
    )
    
    report = compute_power_report(pins)
    
    thermal_info = report.raw.get("thermal", {})
    thermal_raw = report.raw.get("thermal_raw", {})
    
    print(f"   Δn_Kerr: {thermal_info.get('delta_n_kerr', 0):.2e}")
    print(f"   Δn_thermal: {thermal_info.get('delta_n_thermal', 0):.2e}")
    print(f"   Thermal ratio: {thermal_info.get('thermal_ratio', 0):.2f}")
    print(f"   Thermal flag: {thermal_info.get('thermal_flag', 'unknown')}")
    
    if thermal_raw:
        print(f"\n   Raw debug values:")
        print(f"      tau_th_s: {thermal_raw.get('tau_th_s', 0):.2e}")
        print(f"      drift_unscaled: {thermal_raw.get('drift_unscaled', 0):.2e}")
        print(f"      drift_clamped: {thermal_raw.get('drift_clamped', 0):.2e}")
        print(f"      thermal_scale: {thermal_raw.get('thermal_scale', 0)}")
    
    print("\n" + "=" * 60)
    print("DIAGNOSIS:")
    
    expected_ratio = delta_n_thermal_correct / delta_n_kerr
    actual_ratio = thermal_info.get('thermal_ratio', 0)
    
    if abs(actual_ratio - expected_ratio) < 0.01:
        print("✓ Thermal calculation is CORRECT")
    else:
        print("✗ Thermal calculation has a BUG")
        print(f"  Expected ratio: {expected_ratio:.4f}")
        print(f"  Actual ratio: {actual_ratio:.2f}")
        
        # Check if it matches the wrong calculation
        wrong_ratio = delta_n_thermal_wrong / delta_n_kerr
        if abs(actual_ratio - wrong_ratio) < 100:
            print("\n  The bug appears to be a unit conversion issue!")
            print("  tau_thermal_ns is being used directly as seconds")
            print("  instead of being converted from ns to s")

if __name__ == "__main__":
    trace_thermal_calculation()
