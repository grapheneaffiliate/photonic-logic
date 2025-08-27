#!/usr/bin/env python
"""Test thermal calculation in cascade command context."""

import json
from plogic.controller import PhotonicController
from plogic.materials.platforms import MaterialPlatform

def test_cascade_thermal():
    """Test thermal calculation matches between direct and cascade."""
    
    # Create controller with AlGaAs platform
    platform = MaterialPlatform.from_name("AlGaAs")
    controller = PhotonicController(
        wavelength_nm=platform.wavelength_nm,
        n2_m2_per_W=platform.n2_m2_per_W,
        Aeff_um2=platform.Aeff_um2,
        platform_loss_dB_cm=platform.loss_dB_cm,
        dn_dT_per_K=platform.dn_dT_per_K,
        tau_thermal_ns=platform.tau_thermal_ns,
        coupling_eta=0.8,
        link_length_um=100.0,
        L_eff_um=10.0,
        worst_off_norm=0.01,
        extinction_target_dB=21.0,
        include_2pa=platform.include_2pa,
        beta_2pa_m_per_W=platform.beta_2pa_m_per_W,
    )
    
    # Run cascade with same parameters as CLI
    P_high_mW = 0.1
    pulse_ns = 0.3
    
    # Get power report
    power_report = controller.get_power_report(
        P_high_mW=P_high_mW,
        pulse_ns=pulse_ns,
        auto_timing=False
    )
    
    print(f"Testing AlGaAs cascade thermal at {P_high_mW} mW, {pulse_ns} ns")
    print("=" * 60)
    print(f"Energy per operation: {power_report['energetics']['E_op_fJ']} fJ")
    print(f"Delta n_kerr: {power_report['thermal']['delta_n_kerr']:.2e}")
    print(f"Delta n_thermal: {power_report['thermal']['delta_n_thermal']:.2e}")
    print(f"Thermal ratio: {power_report['thermal']['thermal_ratio']:.4f}")
    print(f"Thermal flag: {power_report['thermal']['thermal_flag']}")
    
    # Check if values match expectations
    expected_energy = 30.0
    expected_thermal_ratio = 0.058
    expected_flag = "ok"
    
    energy_match = abs(power_report['energetics']['E_op_fJ'] - expected_energy) < 0.1
    flag_match = power_report['thermal']['thermal_flag'] == expected_flag
    
    print(f"\nExpected vs Actual:")
    print(f"Energy: {expected_energy} fJ vs {power_report['energetics']['E_op_fJ']} fJ")
    print(f"Thermal ratio: ~{expected_thermal_ratio} vs {power_report['thermal']['thermal_ratio']:.4f}")
    print(f"Thermal flag: {expected_flag} vs {power_report['thermal']['thermal_flag']}")
    
    print(f"\nResults:")
    print(f"✅ Energy calculation: {'PASS' if energy_match else 'FAIL'}")
    print(f"✅ Thermal flag: {'PASS' if flag_match else 'FAIL'}")
    
    if power_report['thermal']['thermal_ratio'] < 0.01:
        print(f"\n⚠️ WARNING: Thermal ratio is extremely small ({power_report['thermal']['thermal_ratio']:.2e})")
        print(f"   Expected around {expected_thermal_ratio}")
        print(f"   This suggests the calculation may be using different parameters")

if __name__ == "__main__":
    test_cascade_thermal()
