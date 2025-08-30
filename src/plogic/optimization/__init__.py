"""
Photonic Logic Optimization Module

This module provides DANTE integration for automated photonic circuit optimization.
Includes multi-objective optimization for energy, cascade depth, thermal safety, and fabrication feasibility.
Level 4 system-level optimization for production-ready photonic AI accelerator design.
"""

from .photonic_objectives import (
    PhotonicEnergyOptimizer,
    PhotonicMultiObjective,
    PhotonicCascadeOptimizer,
    PhotonicThermalOptimizer,
)

from .accelerator_system import (
    PhotonicAcceleratorOptimizer,
    optimize_photonic_accelerator,
    generate_fab_ready_specs,
    export_gds_parameters,
    generate_test_patterns,
    generate_compiler_config,
)

from .manufacturing_constraints import (
    ProcessVariationModel,
    YieldModel,
    ProcessCornerModel,
    ReliabilityModel,
    FoundryConstraints,
    TestabilityModel,
    run_manufacturing_analysis,
    export_manufacturing_specs,
)

from .thermal_cosimulation import (
    ThermalProperties,
    HeatSourceModel,
    ThermalSimulator,
    ThermalOptimizer,
    COMLOLInterface,
    run_thermal_analysis,
    export_thermal_specs,
    validate_thermal_design,
    generate_thermal_test_plan,
)

__all__ = [
    # Component-level optimization
    "PhotonicEnergyOptimizer",
    "PhotonicMultiObjective", 
    "PhotonicCascadeOptimizer",
    "PhotonicThermalOptimizer",
    
    # Level 4 system-level optimization
    "PhotonicAcceleratorOptimizer",
    "optimize_photonic_accelerator",
    "generate_fab_ready_specs",
    "export_gds_parameters",
    "generate_test_patterns",
    "generate_compiler_config",
    
    # Manufacturing constraints
    "ProcessVariationModel",
    "YieldModel",
    "ProcessCornerModel",
    "ReliabilityModel",
    "FoundryConstraints",
    "TestabilityModel",
    "run_manufacturing_analysis",
    "export_manufacturing_specs",
    
    # Thermal co-simulation
    "ThermalProperties",
    "HeatSourceModel",
    "ThermalSimulator",
    "ThermalOptimizer",
    "COMLOLInterface",
    "run_thermal_analysis",
    "export_thermal_specs",
    "validate_thermal_design",
    "generate_thermal_test_plan",
]
