"""
Shared statistics extraction utilities for cascade simulations.

This module provides consistent statistics extraction across demo and cascade commands,
ensuring harmonized extinction ratio calculations for realistic engineering assessment.
"""

from typing import Any, Dict


def extract_cascade_statistics(
    cascade_results: Dict[str, Any], 
    extinction_mode: str = "realistic"
) -> Dict[str, float]:
    """
    Extract ON/OFF statistics from cascade simulation results.
    
    Args:
        cascade_results: Dictionary from test_cascade() containing gate simulation results
        extinction_mode: "realistic" (measured from simulation) or "idealized" (theoretical floor)
    
    Returns:
        Dictionary containing:
        - min_on_level: Minimum signal level for logic-1 states
        - max_off_level: Maximum signal level for logic-0 states  
        - worst_off_norm: Normalized worst-case OFF level (max_off/min_on)
        
    Notes:
        - "realistic" mode: Uses actual simulation results for engineering assessment
        - "idealized" mode: Uses theoretical floor (1e-12) for presentation purposes
        - Realistic mode provides accurate extinction ratio for fabrication planning
        - Idealized mode shows theoretical best-case performance for demonstrations
    """
    if extinction_mode not in ["realistic", "idealized"]:
        raise ValueError(f"extinction_mode must be 'realistic' or 'idealized', got '{extinction_mode}'")
    
    if extinction_mode == "idealized":
        # Idealized mode: Use theoretical floors for presentation
        return {
            "min_on_level": 1.0,
            "max_off_level": 1e-12,  # Theoretical floor
            "worst_off_norm": 1e-12,  # Perfect extinction
            "extinction_mode": "idealized"
        }
    
    # Realistic mode: Extract measured statistics from simulation
    min_on_global = float("inf")
    max_off_global = 0.0
    
    for gate_name, gate_data in cascade_results.items():
        if "details" in gate_data and "logic_out" in gate_data:
            for detail, logic_out in zip(gate_data["details"], gate_data["logic_out"]):
                signal = detail.get("signal", 0.0)
                if logic_out == 1:
                    min_on_global = min(min_on_global, signal)
                else:
                    max_off_global = max(max_off_global, signal)
    
    # Handle edge cases
    if min_on_global == float("inf"):
        min_on_global = 1.0
    
    # Calculate normalized worst-case OFF level
    worst_off_norm = max_off_global / max(min_on_global, 1e-30)
    
    return {
        "min_on_level": min_on_global,
        "max_off_level": max_off_global,
        "worst_off_norm": worst_off_norm,
        "extinction_mode": "realistic"
    }


def validate_extinction_mode_flags(realistic_extinction: bool, idealized_extinction: bool) -> str:
    """
    Validate and resolve extinction mode flags.
    
    Args:
        realistic_extinction: Use measured statistics from simulation
        idealized_extinction: Use theoretical floor for presentations
        
    Returns:
        Resolved extinction mode: "realistic" or "idealized"
        
    Raises:
        ValueError: If both flags are True or both are False
    """
    if realistic_extinction and idealized_extinction:
        raise ValueError("Cannot use both --realistic-extinction and --idealized-extinction")
    
    if not realistic_extinction and not idealized_extinction:
        # Default to realistic for engineering assessment
        return "realistic"
    
    return "realistic" if realistic_extinction else "idealized"


def format_extinction_summary(stats: Dict[str, float], target_dB: float = 21.0) -> str:
    """
    Format extinction ratio summary for display.
    
    Args:
        stats: Statistics dictionary from extract_cascade_statistics()
        target_dB: Target extinction ratio in dB
        
    Returns:
        Formatted string summarizing extinction performance
    """
    worst_off_norm = stats["worst_off_norm"]
    extinction_mode = stats.get("extinction_mode", "unknown")
    
    if worst_off_norm <= 0:
        contrast_dB = float("inf")
    else:
        contrast_dB = 10.0 * math.log10(1.0 / max(worst_off_norm, 1e-30))
        contrast_dB = min(contrast_dB, 300.0)  # Cap for display
    
    meets_target = worst_off_norm <= (10.0 ** (-target_dB / 10.0))
    
    return (
        f"Extinction Analysis ({extinction_mode} mode):\n"
        f"  Floor contrast: {contrast_dB:.1f} dB\n"
        f"  Target: {target_dB:.1f} dB\n"
        f"  Meets target: {meets_target}\n"
        f"  Worst OFF level: {worst_off_norm:.2e}"
    )


# Import math for log calculations
import math
