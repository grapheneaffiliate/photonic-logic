"""
Regression tests for thermal calculation fixes.

This test ensures that the thermal calculation bug that caused
thermal ratios of 1,666,666.67 never returns. The correct thermal
ratio for optimized AlGaAs parameters should be around 0.45.
"""

import pytest
from plogic.analysis.power import PowerInputs, compute_power_report
from plogic.materials.platforms import PlatformDB


def test_algaas_thermal_calculation_regression():
    """
    Regression test: Ensure AlGaAs thermal calculation is correct.
    
    This test prevents the return of the thermal calculation bug where
    thermal ratios were incorrectly calculated as 1,666,666.67 instead
    of the correct value around 0.45.
    """
    # Get AlGaAs platform
    pdb = PlatformDB()
    platform = pdb.get("AlGaAs")
    
    # Use the optimized parameters that achieve 33-stage cascade
    cfg = PowerInputs(
        wavelength_nm=1550,
        platform_loss_dB_cm=platform.fabrication.loss_dB_per_cm,
        coupling_eta=0.98,  # Optimized coupling
        link_length_um=60.0,  # Optimized link length
        fanout=1,
        P_high_mW=0.06,  # Optimized ultra-low power
        pulse_ns=1.4,  # Optimized pulse width
        threshold_norm=0.5,
        worst_off_norm=1e-12,
        extinction_target_dB=21.0,
        n2_m2_per_W=platform.nonlinear.n2_m2_per_W,
        Aeff_um2=platform.nonlinear.Aeff_um2_default,
        dn_dT_per_K=platform.thermal.dn_dT_per_K,
        tau_thermal_ns=platform.thermal.tau_thermal_ns,
        thermal_scale=platform.thermal.thermal_scale,
        L_eff_um=10.0,
        include_2pa=platform.flags.tpa_present_at_1550,
        beta_2pa_m_per_W=platform.nonlinear.beta_2pa_m_per_W,
        auto_timing=False
    )
    
    # Compute power report
    report = compute_power_report(cfg)
    
    # Regression assertions
    assert report.thermal_ratio is not None, "Thermal ratio should be calculated"
    assert report.thermal_flag == "ok", "Thermal status should be 'ok'"
    
    # The key regression test: thermal ratio should be around 0.45, NOT 1,666,666.67
    assert 0.3 < report.thermal_ratio < 0.6, (
        f"Thermal ratio {report.thermal_ratio} is outside expected range. "
        f"This may indicate the thermal calculation bug has returned!"
    )
    
    # Additional safety checks
    assert report.thermal_ratio < 2.0, (
        f"Thermal ratio {report.thermal_ratio} exceeds danger threshold. "
        f"Expected safe operation with ratio < 2.0"
    )
    
    # Verify the 33-stage cascade capability
    assert report.max_depth_meeting_thresh >= 30, (
        f"Cascade depth {report.max_depth_meeting_thresh} is below expected 30+ stages. "
        f"This indicates performance regression."
    )
    
    # Verify ultra-low energy operation
    assert report.E_op_fJ < 100, (
        f"Energy per operation {report.E_op_fJ} fJ is higher than expected < 100 fJ"
    )


def test_thermal_scale_factor_is_unity():
    """
    Regression test: Ensure thermal scale factor defaults to 1.0.
    
    The bug was caused by a legacy calibration that set k_th = 2.85e-9
    instead of the correct default of 1.0.
    """
    pdb = PlatformDB()
    platform = pdb.get("AlGaAs")
    
    cfg = PowerInputs(
        wavelength_nm=1550,
        platform_loss_dB_cm=platform.fabrication.loss_dB_per_cm,
        coupling_eta=0.8,
        link_length_um=50.0,
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
        thermal_scale=None,  # Should default to 1.0
        L_eff_um=10.0,
        include_2pa=False,
        beta_2pa_m_per_W=0.0,
        auto_timing=False
    )
    
    report = compute_power_report(cfg)
    
    # Check that thermal scale is 1.0 in the raw debug data
    thermal_raw = report.raw.get("thermal_raw", {})
    thermal_scale = thermal_raw.get("thermal_scale", None)
    
    assert thermal_scale == 1.0, (
        f"Thermal scale factor is {thermal_scale}, expected 1.0. "
        f"Legacy calibration bug may have returned!"
    )


def test_demo_command_thermal_safety():
    """
    Integration test: Verify demo command produces thermally safe results.
    
    This ensures the demo command with default AlGaAs parameters
    produces safe thermal operation.
    """
    from plogic.materials.platforms import PlatformDB
    
    # Simulate the demo command's parameter resolution
    pdb = PlatformDB()
    platform = pdb.get("AlGaAs")
    
    # Demo command defaults for AlGaAs (from cli.py lines 620-635)
    demo_defaults = {
        'P_high_mW': 0.06,      # Optimized for 33-stage cascade
        'pulse_ns': 1.4,        # Optimized for thermal safety
        'coupling_eta': 0.98,   # High efficiency coupling
        'link_length_um': 60.0  # Optimized link length
    }
    
    cfg = PowerInputs(
        wavelength_nm=platform.default_wavelength_nm,
        platform_loss_dB_cm=platform.fabrication.loss_dB_per_cm,
        coupling_eta=demo_defaults['coupling_eta'],
        link_length_um=demo_defaults['link_length_um'],
        fanout=1,
        P_high_mW=demo_defaults['P_high_mW'],
        pulse_ns=demo_defaults['pulse_ns'],
        threshold_norm=0.5,
        worst_off_norm=1e-12,
        extinction_target_dB=21.0,
        n2_m2_per_W=platform.nonlinear.n2_m2_per_W,
        Aeff_um2=platform.nonlinear.Aeff_um2_default,
        dn_dT_per_K=platform.thermal.dn_dT_per_K,
        tau_thermal_ns=platform.thermal.tau_thermal_ns,
        include_2pa=platform.flags.tpa_present_at_1550,
        beta_2pa_m_per_W=platform.nonlinear.beta_2pa_m_per_W,
        auto_timing=False
    )
    
    report = compute_power_report(cfg)
    
    # Verify demo command produces safe thermal operation
    assert report.thermal_flag == "ok", (
        f"Demo command should produce thermally safe operation, got '{report.thermal_flag}'"
    )
    
    # Verify 33-stage capability
    assert report.max_depth_meeting_thresh >= 30, (
        f"Demo command should achieve 30+ stage cascade, got {report.max_depth_meeting_thresh}"
    )


if __name__ == "__main__":
    # Run the tests directly
    test_algaas_thermal_calculation_regression()
    test_thermal_scale_factor_is_unity()
    test_demo_command_thermal_safety()
    print("✅ All thermal regression tests passed!")
