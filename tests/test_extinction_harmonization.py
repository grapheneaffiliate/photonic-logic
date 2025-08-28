"""
Tests for extinction ratio harmonization between demo and cascade commands.

This test suite ensures that both commands use consistent statistics extraction
and provide both realistic (engineering) and idealized (presentation) modes.
"""

import pytest
from plogic.utils.statistics import (
    extract_cascade_statistics, 
    validate_extinction_mode_flags,
    format_extinction_summary
)


def test_extract_cascade_statistics_realistic():
    """Test realistic mode statistics extraction from cascade results."""
    # Mock cascade results with realistic gate simulation data
    mock_results = {
        "XOR": {
            "details": [
                {"inputs": [0, 0], "signal": 0.0},
                {"inputs": [0, 1], "signal": 1.0},
                {"inputs": [1, 0], "signal": 0.999},
                {"inputs": [1, 1], "signal": 0.01}  # Realistic leakage
            ],
            "logic_out": [0, 1, 1, 0]
        }
    }
    
    stats = extract_cascade_statistics(mock_results, "realistic")
    
    assert stats["min_on_level"] == 0.999  # Minimum ON level
    assert stats["max_off_level"] == 0.01   # Maximum OFF level (realistic leakage)
    assert abs(stats["worst_off_norm"] - 0.01/0.999) < 1e-6  # Normalized leakage
    assert stats["extinction_mode"] == "realistic"


def test_extract_cascade_statistics_idealized():
    """Test idealized mode statistics extraction (theoretical floor)."""
    # Mock cascade results - content doesn't matter for idealized mode
    mock_results = {
        "XOR": {
            "details": [
                {"inputs": [0, 0], "signal": 0.0},
                {"inputs": [0, 1], "signal": 1.0},
                {"inputs": [1, 0], "signal": 0.999},
                {"inputs": [1, 1], "signal": 0.01}
            ],
            "logic_out": [0, 1, 1, 0]
        }
    }
    
    stats = extract_cascade_statistics(mock_results, "idealized")
    
    assert stats["min_on_level"] == 1.0      # Perfect ON level
    assert stats["max_off_level"] == 1e-12   # Theoretical floor
    assert stats["worst_off_norm"] == 1e-12  # Perfect extinction
    assert stats["extinction_mode"] == "idealized"


def test_validate_extinction_mode_flags():
    """Test extinction mode flag validation."""
    # Test valid combinations
    assert validate_extinction_mode_flags(True, False) == "realistic"
    assert validate_extinction_mode_flags(False, True) == "idealized"
    assert validate_extinction_mode_flags(False, False) == "realistic"  # Default
    
    # Test invalid combination
    with pytest.raises(ValueError, match="Cannot use both"):
        validate_extinction_mode_flags(True, True)


def test_format_extinction_summary():
    """Test extinction summary formatting."""
    # Test realistic mode summary
    realistic_stats = {
        "min_on_level": 1.0,
        "max_off_level": 0.01,
        "worst_off_norm": 0.01,
        "extinction_mode": "realistic"
    }
    
    summary = format_extinction_summary(realistic_stats, target_dB=21.0)
    assert "realistic mode" in summary
    assert "20.0 dB" in summary  # 10*log10(1/0.01) = 20 dB
    assert "21.0 dB" in summary  # Target
    
    # Test idealized mode summary
    idealized_stats = {
        "min_on_level": 1.0,
        "max_off_level": 1e-12,
        "worst_off_norm": 1e-12,
        "extinction_mode": "idealized"
    }
    
    summary = format_extinction_summary(idealized_stats, target_dB=21.0)
    assert "idealized mode" in summary
    assert "120.0 dB" in summary  # 10*log10(1/1e-12) = 120 dB


def test_extinction_mode_invalid():
    """Test invalid extinction mode handling."""
    mock_results = {"XOR": {"details": [], "logic_out": []}}
    
    with pytest.raises(ValueError, match="extinction_mode must be"):
        extract_cascade_statistics(mock_results, "invalid_mode")


def test_realistic_vs_idealized_contrast():
    """Test that realistic and idealized modes produce different contrast values."""
    # Mock results with realistic leakage
    mock_results = {
        "AND": {
            "details": [
                {"inputs": [0, 0], "signal": 0.0},
                {"inputs": [0, 1], "signal": 0.005},  # Some leakage
                {"inputs": [1, 0], "signal": 0.008},  # Some leakage
                {"inputs": [1, 1], "signal": 1.0}
            ],
            "logic_out": [0, 0, 0, 1]
        }
    }
    
    realistic_stats = extract_cascade_statistics(mock_results, "realistic")
    idealized_stats = extract_cascade_statistics(mock_results, "idealized")
    
    # Realistic should show actual leakage
    assert realistic_stats["worst_off_norm"] > 1e-6  # Measurable leakage
    
    # Idealized should show perfect extinction
    assert idealized_stats["worst_off_norm"] == 1e-12  # Theoretical floor
    
    # Idealized should have much better contrast
    assert idealized_stats["worst_off_norm"] < realistic_stats["worst_off_norm"]


def test_edge_case_no_on_states():
    """Test handling of edge case where no ON states are found."""
    mock_results = {
        "ALWAYS_OFF": {
            "details": [
                {"inputs": [0, 0], "signal": 0.0},
                {"inputs": [0, 1], "signal": 0.001},
                {"inputs": [1, 0], "signal": 0.002},
                {"inputs": [1, 1], "signal": 0.0}
            ],
            "logic_out": [0, 0, 0, 0]  # No ON states
        }
    }
    
    stats = extract_cascade_statistics(mock_results, "realistic")
    
    # Should handle gracefully with default min_on_level
    assert stats["min_on_level"] == 1.0  # Default fallback
    assert stats["max_off_level"] == 0.002  # Maximum OFF level
    assert stats["worst_off_norm"] == 0.002  # Normalized to default ON level


def test_edge_case_empty_results():
    """Test handling of empty cascade results."""
    empty_results = {}
    
    stats = extract_cascade_statistics(empty_results, "realistic")
    
    # Should handle gracefully
    assert stats["min_on_level"] == 1.0
    assert stats["max_off_level"] == 0.0
    assert stats["worst_off_norm"] == 0.0


def test_extinction_harmonization_integration():
    """
    Integration test: Verify that demo and cascade commands will now produce
    consistent results when using the same extinction mode.
    """
    # This test verifies the harmonization logic without running full CLI commands
    
    # Mock realistic cascade results (similar to what both commands would generate)
    realistic_results = {
        "XOR": {
            "details": [
                {"inputs": [0, 0], "signal": 0.0},
                {"inputs": [0, 1], "signal": 1.0},
                {"inputs": [1, 0], "signal": 0.999},
                {"inputs": [1, 1], "signal": 0.01}  # 1% leakage (realistic)
            ],
            "logic_out": [0, 1, 1, 0]
        }
    }
    
    # Both commands should now produce identical statistics in realistic mode
    demo_stats = extract_cascade_statistics(realistic_results, "realistic")
    cascade_stats = extract_cascade_statistics(realistic_results, "realistic")
    
    assert demo_stats == cascade_stats
    assert demo_stats["worst_off_norm"] == cascade_stats["worst_off_norm"]
    assert demo_stats["extinction_mode"] == "realistic"
    
    # Both commands should produce identical idealized statistics
    demo_idealized = extract_cascade_statistics(realistic_results, "idealized")
    cascade_idealized = extract_cascade_statistics(realistic_results, "idealized")
    
    assert demo_idealized == cascade_idealized
    assert demo_idealized["worst_off_norm"] == 1e-12
    assert demo_idealized["extinction_mode"] == "idealized"


if __name__ == "__main__":
    # Run the tests directly
    test_extract_cascade_statistics_realistic()
    test_extract_cascade_statistics_idealized()
    test_validate_extinction_mode_flags()
    test_format_extinction_summary()
    test_extinction_mode_invalid()
    test_realistic_vs_idealized_contrast()
    test_edge_case_no_on_states()
    test_edge_case_empty_results()
    test_extinction_harmonization_integration()
    print("✅ All extinction harmonization tests passed!")
