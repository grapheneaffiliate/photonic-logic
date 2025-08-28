"""Test extinction ratio boundary conditions and edge cases."""
import pytest
from plogic.analysis import PowerInputs, compute_power_report


def test_er_boundary_exact():
    """Test ER boundary at exactly 20 dB and 21 dB targets."""
    
    # Base configuration with worst_off_norm = 0.01 (exactly 20 dB)
    base_config = {
        "wavelength_nm": 1550.0,
        "platform_loss_dB_cm": 0.1,
        "coupling_eta": 0.8,
        "link_length_um": 50.0,
        "fanout": 1,
        "pulse_ns": 1.0,
        "P_high_mW": 1.0,
        "threshold_norm": 0.5,
        "worst_off_norm": 0.01,  # Exactly 20 dB contrast
        "er_epsilon": 1e-12,
    }
    
    # Test 1: Target = 20 dB, should PASS (exactly meets)
    pins_20dB = PowerInputs(**base_config, extinction_target_dB=20.0)
    rep_20dB = compute_power_report(pins_20dB)
    assert rep_20dB.meets_extinction is True, "Should meet 20 dB target with 0.01 worst_off"
    assert abs(rep_20dB.raw["contrast_breakdown"]["margin_dB"]) < 0.01, "Should have ~0 margin"
    
    # Test 2: Target = 21 dB, should FAIL (misses by 1 dB)
    pins_21dB = PowerInputs(**base_config, extinction_target_dB=21.0)
    rep_21dB = compute_power_report(pins_21dB)
    assert rep_21dB.meets_extinction is False, "Should fail 21 dB target with 0.01 worst_off"
    assert rep_21dB.raw["contrast_breakdown"]["margin_dB"] < 0, "Should have negative margin"
    
    # Test 3: Target = 19.9 dB, should PASS (exceeds by 0.1 dB)
    pins_19_9dB = PowerInputs(**base_config, extinction_target_dB=19.9)
    rep_19_9dB = compute_power_report(pins_19_9dB)
    assert rep_19_9dB.meets_extinction is True, "Should meet 19.9 dB target"
    assert rep_19_9dB.raw["contrast_breakdown"]["margin_dB"] > 0, "Should have positive margin"


def test_er_epsilon_tolerance():
    """Test that er_epsilon provides proper tolerance."""
    
    base_config = {
        "wavelength_nm": 1550.0,
        "platform_loss_dB_cm": 0.1,
        "coupling_eta": 0.8,
        "link_length_um": 50.0,
        "fanout": 1,
        "pulse_ns": 1.0,
        "P_high_mW": 1.0,
        "threshold_norm": 0.5,
        "worst_off_norm": 0.0105,  # Slightly worse than 0.01
        "extinction_target_dB": 20.0,
    }
    
    # Without epsilon tolerance - should fail
    pins_no_eps = PowerInputs(**base_config, er_epsilon=0)
    rep_no_eps = compute_power_report(pins_no_eps)
    assert rep_no_eps.meets_extinction is False, "Should fail without epsilon tolerance"
    
    # With generous epsilon - should pass
    pins_with_eps = PowerInputs(**base_config, er_epsilon=0.001)
    rep_with_eps = compute_power_report(pins_with_eps)
    assert rep_with_eps.meets_extinction is True, "Should pass with epsilon tolerance"


def test_er_extreme_values():
    """Test ER calculation with extreme values."""
    
    # Test near-zero worst_off (should not cause infinity)
    pins_zero = PowerInputs(
        wavelength_nm=1550.0,
        platform_loss_dB_cm=0.1,
        coupling_eta=0.8,
        link_length_um=50.0,
        fanout=1,
        pulse_ns=1.0,
        P_high_mW=1.0,
        threshold_norm=0.5,
        worst_off_norm=1e-15,  # Near-zero
        extinction_target_dB=30.0,
        er_epsilon=1e-12,
    )
    
    rep_zero = compute_power_report(pins_zero)
    # Should handle gracefully without infinity
    assert rep_zero.raw["contrast_breakdown"]["floor_contrast_dB"] <= 300, "Should cap at reasonable value"
    assert rep_zero.raw["contrast_breakdown"]["floor_contrast_dB"] > 100, "Should show high contrast"
    
    # Test high worst_off (poor extinction)
    pins_high = PowerInputs(
        wavelength_nm=1550.0,
        platform_loss_dB_cm=0.1,
        coupling_eta=0.8,
        link_length_um=50.0,
        fanout=1,
        pulse_ns=1.0,
        P_high_mW=1.0,
        threshold_norm=0.5,
        worst_off_norm=0.5,  # Very poor (3 dB)
        extinction_target_dB=20.0,
        er_epsilon=1e-12,
    )
    
    rep_high = compute_power_report(pins_high)
    assert rep_high.meets_extinction is False, "Should fail with poor extinction"
    assert rep_high.raw["contrast_breakdown"]["floor_contrast_dB"] < 10, "Should show poor contrast"


def test_er_consistency():
    """Test that ER calculations are self-consistent."""
    
    pins = PowerInputs(
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
        er_epsilon=1e-12,
    )
    
    rep = compute_power_report(pins)
    
    # Verify internal consistency
    floor_dB = rep.raw["contrast_breakdown"]["floor_contrast_dB"]
    target_dB = rep.raw["contrast_breakdown"]["target_contrast_dB"]
    margin_dB = rep.raw["contrast_breakdown"]["margin_dB"]
    
    assert abs((floor_dB - target_dB) - margin_dB) < 0.001, "Margin should equal floor - target"
    
    # Verify worst_off_norm matches floor_contrast_dB
    import math
    calculated_floor = 10 * math.log10(1.0 / max(pins.worst_off_norm, 1e-30))
    assert abs(floor_dB - calculated_floor) < 0.001, "Floor dB should match worst_off calculation"
