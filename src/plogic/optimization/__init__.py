"""
Photonic Logic Optimization Module

This module provides DANTE integration for automated photonic circuit optimization.
Includes multi-objective optimization for energy, cascade depth, thermal safety, and fabrication feasibility.
"""

from .photonic_objectives import (
    PhotonicEnergyOptimizer,
    PhotonicMultiObjective,
    PhotonicCascadeOptimizer,
    PhotonicThermalOptimizer,
)

__all__ = [
    "PhotonicEnergyOptimizer",
    "PhotonicMultiObjective", 
    "PhotonicCascadeOptimizer",
    "PhotonicThermalOptimizer",
]
