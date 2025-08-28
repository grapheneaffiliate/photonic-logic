#!/usr/bin/env python
"""Direct test of optimized cascade configuration."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from plogic.materials.platforms import PlatformDB
from plogic.analysis.power import PowerInputs, compute_power_report

def test_optimized_cascade():
    """Test that optimized parameters achieve cascade depth >= 30."""
    
    print("Testing optimized AlGaAs cascade configuration...")
    print("-" * 50)
    
    # Get AlGaAs platform
    db = PlatformDB()
    platform = db.get("AlGaAs")
    print(f"Platform: {platform.name}")
    print(f"  n2: {platform.nonlinear.n2_m2_per_W:.2e} m²/W")
    print(f"  Loss: {platform.fabrication.loss_dB_per_cm} dB/cm")
    print(f"  dn/dT: {platform.thermal.dn_dT_per_K:.2e} /K")
    
    # Set up optimized parameters
    cfg = PowerInputs(
        wavelength_nm=1550,
        platform_loss_dB_cm=platform.fabrication.loss_dB_per_cm,
        coupling_eta=0.98,
        link_length_um=60.0,
        fanout=1,
        P_high_mW=0.06,
        pulse_ns=1.4,
        threshold_norm=0.5,
        worst_off_norm=1e-12,
        extinction_target_dB=21.0,
        n2_m2_per_W=platform.nonlinear.n2_m2_per_W,
        Aeff_um2=platform.nonlinear.Aeff_um2_default,
        dn_dT_per_K=platform.thermal.dn_dT_per_K,
        tau_thermal_ns=platform.thermal.tau_thermal_ns,
        thermal_scale=platform.thermal.thermal_scale,
        L_eff_um=10.0,  # Default effective length
        include_2pa=platform.flags.tpa_present_at_1550,
        beta_2pa_m_per_W=platform.nonlinear.beta_2pa_m_per_W,
        auto_timing=False
    )
    
    print("\nOptimized parameters:")
    print(f"  P_high: {cfg.P_high_mW} mW")
    print(f"  Pulse: {cfg.pulse_ns} ns")
    print(f"  Coupling η: {cfg.coupling_eta}")
    print(f"  Link length: {cfg.link_length_um} µm")
    
    # Compute power report
    report = compute_power_report(cfg)
    
    print("\nResults:")
    print("-" * 30)
    print(f"✓ Cascade depth: {report.max_depth_meeting_thresh} stages")
    print(f"  Per-stage transmittance: {report.per_stage_transmittance:.4f}")
    print(f"  P_threshold: {report.P_threshold_mW:.4f} mW")
    
    print(f"\n✓ Energy: {report.E_op_fJ:.1f} fJ/op")
    print(f"  Photons/op: {report.photons_per_op:.1e}")
    
    if report.thermal_ratio is not None:
        print(f"\n✓ Thermal: {report.thermal_flag}")
        print(f"  Thermal ratio: {report.thermal_ratio:.4f}")
    
    # Check success
    print("=" * 50)
    if report.max_depth_meeting_thresh >= 30:
        print(f"🎉 SUCCESS: Achieved {report.max_depth_meeting_thresh} stages (target: ≥30)")
        if report.thermal_flag == "ok":
            print("✅ Thermal safety confirmed")
        return True
    else:
        print(f"❌ FAILED: Only achieved {report.max_depth_meeting_thresh} stages (target: ≥30)")
        return False

if __name__ == "__main__":
    success = test_optimized_cascade()
    sys.exit(0 if success else 1)
