#!/usr/bin/env python
"""Direct test of cascade depth calculation with optimized parameters."""

import json
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from plogic.analysis import PowerInputs, compute_power_report
from plogic.materials import PlatformDB

def test_cascade_depth():
    """Test cascade depth with optimized AlGaAs parameters."""
    
    print("Testing cascade depth with optimized parameters...")
    print("-" * 50)
    
    # Load AlGaAs platform
    pdb = PlatformDB()
    platform = pdb.get("AlGaAs")
    
    print(f"Platform: {platform.name}")
    print(f"  n2: {platform.nonlinear.n2_m2_per_W:.2e} m²/W")
    print(f"  Loss: {platform.fabrication.loss_dB_per_cm} dB/cm")
    print(f"  dn/dT: {platform.thermal.dn_dT_per_K:.2e} /K")
    print()
    
    # Optimized parameters
    P_high_mW = 0.06
    pulse_ns = 1.4
    coupling_eta = 0.98
    link_length_um = 60.0
    
    print("Optimized parameters:")
    print(f"  P_high: {P_high_mW} mW")
    print(f"  Pulse: {pulse_ns} ns")
    print(f"  Coupling η: {coupling_eta}")
    print(f"  Link length: {link_length_um} µm")
    print()
    
    # Build power analysis inputs
    pins = PowerInputs(
        wavelength_nm=platform.default_wavelength_nm,
        platform_loss_dB_cm=platform.fabrication.loss_dB_per_cm,
        coupling_eta=coupling_eta,
        link_length_um=link_length_um,
        fanout=1,
        pulse_ns=pulse_ns,
        P_high_mW=P_high_mW,
        threshold_norm=0.5,
        worst_off_norm=0.01,  # Typical value
        extinction_target_dB=21.0,
        er_epsilon=1e-12,
        n2_m2_per_W=platform.nonlinear.n2_m2_per_W,
        Aeff_um2=platform.nonlinear.Aeff_um2_default,
        dn_dT_per_K=platform.thermal.dn_dT_per_K,
        tau_thermal_ns=platform.thermal.tau_thermal_ns,
        include_2pa=platform.flags.tpa_present_at_1550,
        beta_2pa_m_per_W=platform.nonlinear.beta_2pa_m_per_W,
        auto_timing=False
    )
    
    # Compute power report
    power_rep = compute_power_report(pins)
    
    # Extract results
    cascade_info = power_rep.raw.get("cascade", {})
    max_depth = cascade_info.get("max_depth_meeting_thresh", 0)
    per_stage_T = cascade_info.get("per_stage_transmittance", 0)
    
    thermal_info = power_rep.raw.get("thermal", {})
    thermal_flag = thermal_info.get("thermal_flag", "unknown")
    thermal_ratio = thermal_info.get("thermal_ratio", 0)
    
    energetics = power_rep.raw.get("energetics", {})
    E_op_fJ = energetics.get("E_op_fJ", 0)
    
    print("Results:")
    print("-" * 30)
    print(f"✓ Cascade depth: {max_depth} stages")
    print(f"  Per-stage transmittance: {per_stage_T:.4f}")
    print(f"  P_threshold: {cascade_info.get('P_threshold_mW', 0):.4f} mW")
    print()
    print(f"✓ Energy: {E_op_fJ:.1f} fJ/op")
    print(f"  Photons/op: {energetics.get('photons_per_op', 0):.1e}")
    print()
    print(f"✓ Thermal: {thermal_flag}")
    print(f"  Thermal ratio: {thermal_ratio:.4f}")
    
    # Check if we meet the target
    print()
    print("=" * 50)
    if max_depth >= 30:
        print(f"🎉 SUCCESS: Achieved {max_depth} stages (target: ≥30)")
        return True
    else:
        print(f"❌ FAILED: Only {max_depth} stages (target: ≥30)")
        
        # Debug information
        print("\nDebug info:")
        print(f"  Transmittance per stage: {per_stage_T:.6f}")
        print(f"  Loss factor: {per_stage_T / coupling_eta:.6f}")
        
        # Calculate what cascade depth should be
        import math
        if per_stage_T > 0 and per_stage_T < 1:
            theoretical_depth = math.floor(math.log(0.5) / math.log(per_stage_T))
            print(f"  Theoretical depth (0.5 threshold): {theoretical_depth}")
        
        return False

if __name__ == "__main__":
    success = test_cascade_depth()
    sys.exit(0 if success else 1)
