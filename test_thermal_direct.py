#!/usr/bin/env python3
"""
Direct test of the thermal calculation fix.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from plogic.analysis.power import PowerInputs, compute_power_report

def test_algaas_thermal():
    print("Testing AlGaAs thermal calculation at 0.1 mW, 0.3 ns")
    print("=" * 60)
    
    # AlGaAs parameters from database
    pins = PowerInputs(
        wavelength_nm=1550.0,
        platform_loss_dB_cm=0.5,  # AlGaAs loss
        coupling_eta=0.8,
        link_length_um=50.0,
        fanout=1,
        pulse_ns=0.3,
        P_high_mW=0.1,
        threshold_norm=0.5,
        worst_off_norm=0.0,  # Perfect extinction for test
        extinction_target_dB=21.0,
        n2_m2_per_W=1.5e-17,  # AlGaAs n2
        Aeff_um2=0.5,  # AlGaAs Aeff
        dn_dT_per_K=3.0e-4,  # AlGaAs thermal coefficient
        tau_thermal_ns=60.0,  # AlGaAs thermal time constant
        L_eff_um=10.0,
        include_2pa=True,
        beta_2pa_m_per_W=1.0e-10,  # AlGaAs 2PA
        auto_timing=False
    )
    
    power_rep = compute_power_report(pins)
    
    print(f"Energy per operation: {power_rep.E_op_fJ:.1f} fJ")
    print(f"Delta n_kerr: {power_rep.delta_n_kerr:.2e}")
    print(f"Delta n_thermal: {power_rep.delta_n_thermal:.2e}")
    print(f"Thermal ratio: {power_rep.thermal_ratio:.4f}")
    print(f"Thermal flag: {power_rep.thermal_flag}")
    print()
    
    # Expected results
    expected_energy = 30.0  # fJ
    expected_thermal_flag = "ok"  # Should be ok with new thresholds
    
    print("Expected vs Actual:")
    print(f"Energy: {expected_energy} fJ vs {power_rep.E_op_fJ:.1f} fJ")
    print(f"Thermal flag: {expected_thermal_flag} vs {power_rep.thermal_flag}")
    
    # Check if fix worked
    energy_ok = abs(power_rep.E_op_fJ - expected_energy) < 1.0
    thermal_ok = power_rep.thermal_flag == expected_thermal_flag
    
    print()
    print("Results:")
    print(f"✅ Energy calculation: {'PASS' if energy_ok else 'FAIL'}")
    print(f"✅ Thermal flag: {'PASS' if thermal_ok else 'FAIL'}")
    
    if thermal_ok and energy_ok:
        print()
        print("🎉 SUCCESS: AlGaAs thermal calculation is now working correctly!")
        print("   - 30 fJ energy per operation ✓")
        print("   - Thermal flag shows 'ok' ✓")
        print("   - Ready for production showcase ✓")
    else:
        print()
        print("❌ ISSUE: Thermal calculation still needs work")
        if not energy_ok:
            print(f"   - Energy mismatch: expected {expected_energy}, got {power_rep.E_op_fJ:.1f}")
        if not thermal_ok:
            print(f"   - Thermal flag wrong: expected {expected_thermal_flag}, got {power_rep.thermal_flag}")
            print(f"   - Thermal ratio: {power_rep.thermal_ratio:.4f}")

if __name__ == "__main__":
    test_algaas_thermal()
