import pytest
from plogic.analysis import PowerInputs, compute_power_report


def test_power_inputs_creation():
    pins = PowerInputs(
        wavelength_nm=1550.0,
        platform_loss_dB_cm=0.1,
        coupling_eta=0.8,
        link_length_um=50.0,
        fanout=1,
        P_high_mW=1.0,
        threshold_norm=0.5,
        worst_off_norm=0.01,
        extinction_target_dB=20.0
    )
    assert pins.wavelength_nm == 1550.0
    assert pins.platform_loss_dB_cm == 0.1
    assert pins.coupling_eta == 0.8


def test_basic_power_report():
    pins = PowerInputs(
        wavelength_nm=1550.0,
        platform_loss_dB_cm=0.1,
        coupling_eta=0.8,
        link_length_um=50.0,
        fanout=1,
        pulse_ns=1.0,
        q_factor=1e6,
        P_high_mW=1.0,
        threshold_norm=0.5,
        worst_off_norm=0.01,
        extinction_target_dB=20.0,
        n2_m2_per_W=1e-18,
        Aeff_um2=0.6,
        dn_dT_per_K=1e-4,
        tau_thermal_ns=100.0,
        auto_timing=False
    )
    
    rep = compute_power_report(pins)
    
    # Check basic calculations
    assert rep.E_op_fJ > 0
    assert rep.photons_per_op > 0
    assert rep.per_stage_transmittance > 0
    assert rep.P_threshold_mW == 0.5
    assert rep.max_depth_meeting_thresh >= 0
    assert rep.thermal_flag in ["ok", "caution", "danger"]


def test_auto_timing_from_q():
    pins = PowerInputs(
        wavelength_nm=1550.0,
        platform_loss_dB_cm=0.1,
        coupling_eta=0.8,
        link_length_um=50.0,
        fanout=1,
        q_factor=1e6,
        P_high_mW=1.0,
        threshold_norm=0.5,
        worst_off_norm=0.01,
        extinction_target_dB=20.0,
        auto_timing=True
    )
    
    rep = compute_power_report(pins)
    
    # Should derive timing from Q factor
    assert rep.tau_ph_ns is not None
    assert rep.t_switch_ns > 0
    assert rep.t_switch_ns == 2.0 * rep.tau_ph_ns


def test_thermal_analysis():
    # Test thermal flag transitions
    base_pins = PowerInputs(
        wavelength_nm=1550.0,
        platform_loss_dB_cm=0.1,
        coupling_eta=0.8,
        link_length_um=50.0,
        fanout=1,
        pulse_ns=1.0,
        P_high_mW=0.1,  # Low power
        threshold_norm=0.5,
        worst_off_norm=0.01,
        extinction_target_dB=20.0,
        n2_m2_per_W=1e-18,
        Aeff_um2=0.6,
        dn_dT_per_K=1e-5,  # Low thermal coefficient
        tau_thermal_ns=100.0,
        auto_timing=False
    )
    
    rep_low = compute_power_report(base_pins)
    assert rep_low.thermal_flag == "ok"
    
    # High thermal coefficient should trigger caution/danger
    high_thermal_pins = PowerInputs(
        wavelength_nm=1550.0,
        platform_loss_dB_cm=0.1,
        coupling_eta=0.8,
        link_length_um=50.0,
        fanout=1,
        pulse_ns=10.0,  # Longer pulse
        P_high_mW=2.0,  # Higher power
        threshold_norm=0.5,
        worst_off_norm=0.01,
        extinction_target_dB=20.0,
        n2_m2_per_W=1e-19,  # Weaker Kerr effect
        Aeff_um2=0.6,
        dn_dT_per_K=5e-4,  # Very high thermal coefficient
        tau_thermal_ns=10.0,  # Very fast thermal response
        auto_timing=False
    )
    
    rep_high = compute_power_report(high_thermal_pins)
    # Should trigger caution or danger due to high thermal ratio
    assert rep_high.thermal_ratio > 0.2  # At least in caution range


def test_cascade_depth_calculation():
    # Test cascade depth with different fanout values
    pins_fanout1 = PowerInputs(
        wavelength_nm=1550.0,
        platform_loss_dB_cm=0.1,
        coupling_eta=0.8,
        link_length_um=50.0,
        fanout=1,
        pulse_ns=1.0,
        P_high_mW=1.0,
        threshold_norm=0.5,
        worst_off_norm=0.01,
        extinction_target_dB=20.0
    )
    
    pins_fanout2 = PowerInputs(
        wavelength_nm=1550.0,
        platform_loss_dB_cm=0.1,
        coupling_eta=0.8,
        link_length_um=50.0,
        fanout=2,
        pulse_ns=1.0,
        P_high_mW=1.0,
        threshold_norm=0.5,
        worst_off_norm=0.01,
        extinction_target_dB=20.0
    )
    
    rep1 = compute_power_report(pins_fanout1)
    rep2 = compute_power_report(pins_fanout2)
    
    # Higher fanout should reduce cascade depth
    assert rep1.max_depth_meeting_thresh >= rep2.max_depth_meeting_thresh


def test_extinction_ratio_check():
    # Test extinction ratio validation
    pins_good = PowerInputs(
        wavelength_nm=1550.0,
        platform_loss_dB_cm=0.1,
        coupling_eta=0.8,
        link_length_um=50.0,
        fanout=1,
        pulse_ns=1.0,
        P_high_mW=1.0,
        threshold_norm=0.5,
        worst_off_norm=0.001,  # Good extinction
        extinction_target_dB=20.0
    )
    
    pins_bad = PowerInputs(
        wavelength_nm=1550.0,
        platform_loss_dB_cm=0.1,
        coupling_eta=0.8,
        link_length_um=50.0,
        fanout=1,
        pulse_ns=1.0,
        P_high_mW=1.0,
        threshold_norm=0.5,
        worst_off_norm=0.1,  # Poor extinction
        extinction_target_dB=20.0
    )
    
    rep_good = compute_power_report(pins_good)
    rep_bad = compute_power_report(pins_bad)
    
    assert rep_good.meets_extinction is True
    assert rep_bad.meets_extinction is False


def test_energy_scaling():
    # Test energy scaling with power and timing
    pins_low = PowerInputs(
        wavelength_nm=1550.0,
        platform_loss_dB_cm=0.1,
        coupling_eta=0.8,
        link_length_um=50.0,
        fanout=1,
        pulse_ns=1.0,
        P_high_mW=0.5,
        threshold_norm=0.5,
        worst_off_norm=0.01,
        extinction_target_dB=20.0
    )
    
    pins_high = PowerInputs(
        wavelength_nm=1550.0,
        platform_loss_dB_cm=0.1,
        coupling_eta=0.8,
        link_length_um=50.0,
        fanout=1,
        pulse_ns=1.0,
        P_high_mW=1.0,
        threshold_norm=0.5,
        worst_off_norm=0.01,
        extinction_target_dB=20.0
    )
    
    rep_low = compute_power_report(pins_low)
    rep_high = compute_power_report(pins_high)
    
    # Higher power should give higher energy
    assert rep_high.E_op_fJ > rep_low.E_op_fJ
    assert rep_high.photons_per_op > rep_low.photons_per_op


def test_2pa_inclusion():
    # Test two-photon absorption inclusion
    pins_no_2pa = PowerInputs(
        wavelength_nm=1550.0,
        platform_loss_dB_cm=0.1,
        coupling_eta=0.8,
        link_length_um=50.0,
        fanout=1,
        pulse_ns=1.0,
        P_high_mW=1.0,
        threshold_norm=0.5,
        worst_off_norm=0.01,
        extinction_target_dB=20.0,
        n2_m2_per_W=1e-18,
        Aeff_um2=0.6,
        dn_dT_per_K=1e-4,
        tau_thermal_ns=100.0,
        include_2pa=False,
        beta_2pa_m_per_W=0.0
    )
    
    pins_with_2pa = PowerInputs(
        wavelength_nm=1550.0,
        platform_loss_dB_cm=0.1,
        coupling_eta=0.8,
        link_length_um=50.0,
        fanout=1,
        pulse_ns=1.0,
        P_high_mW=1.0,
        threshold_norm=0.5,
        worst_off_norm=0.01,
        extinction_target_dB=20.0,
        n2_m2_per_W=1e-18,
        Aeff_um2=0.6,
        dn_dT_per_K=1e-4,
        tau_thermal_ns=100.0,
        include_2pa=True,
        beta_2pa_m_per_W=1e-10
    )
    
    rep_no_2pa = compute_power_report(pins_no_2pa)
    rep_with_2pa = compute_power_report(pins_with_2pa)
    
    # Both should complete without error
    assert rep_no_2pa.thermal_flag is not None
    assert rep_with_2pa.thermal_flag is not None
