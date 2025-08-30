"""
Level 4 System-Level Photonic AI Accelerator Optimization

This module implements production-ready optimization for the mobile photonic AI accelerator,
including 4000+ ring arrays, thermal co-simulation, manufacturing constraints, and yield optimization.
"""

import sys
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple, Union
import numpy as np
import json
import warnings
from pathlib import Path

# Add DANTE to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'DANTE'))
from dante.obj_functions import ObjectiveFunction
from dante.utils import Tracker

# Import photonic logic components
from ..controller import ExperimentController, PhotonicMolecule
from ..materials.platforms import PlatformDB


@dataclass
class AcceleratorConfig:
    """Configuration for the photonic AI accelerator system."""
    
    # System architecture
    num_rings: int = 4000
    array_config: Tuple[int, int] = (64, 64)  # 64x64 ring array
    num_lanes: int = 17  # Parallel computational lanes
    num_tiles: int = 4   # 4x4 tile configuration
    
    # Power constraints (mobile)
    total_power_budget_W: float = 2.0  # Hard mobile limit
    laser_power_budget_W: float = 0.53  # 4x QDFB lasers
    ring_tuning_budget_W: float = 0.20  # 4000 rings × 50µW
    sram_power_budget_W: float = 0.35   # 3D-stacked SRAM
    
    # Performance targets
    target_sustained_tops: float = 3.11
    target_token_rate: float = 50.0  # tok/s for 7B model
    target_latency_ms: float = 10.0  # Maximum inference latency
    
    # Manufacturing constraints
    target_yield: float = 0.8  # 80% functional rings
    cd_variation_nm: float = 5.0  # ±5nm critical dimension
    wavelength_drift_pm: float = 5.0  # <5pm drift over 1 hour
    
    # Thermal constraints
    max_die_temp_C: float = 85.0  # Mobile operating temperature
    thermal_gradient_limit: float = 10.0  # °C across die
    heater_power_limit_uW: float = 50.0  # Per ring
    
    # Fabrication process
    platform: str = "AlGaAs"  # AlGaAsOI platform
    process_node: str = "150nm"  # Initial prototyping
    wafer_size_mm: int = 150  # 150mm → 200mm scaling
    
    # Integration specs
    sram_capacity_MB: int = 512  # 3D-stacked SRAM
    sram_bandwidth_TBps: float = 2.0  # Cu-Cu bonding bandwidth
    package_type: str = "InFO_oS"  # TSMC InFO on Substrate


@dataclass
class ManufacturingModel:
    """Manufacturing and yield modeling for the accelerator."""
    
    # Process parameters
    cd_mean_nm: float = 220.0  # Target critical dimension
    cd_std_nm: float = 2.0     # Process variation (3σ = ±6nm)
    sidewall_roughness_nm: float = 1.5  # RMS sidewall roughness
    
    # Yield parameters
    ring_yield_probability: float = 0.85  # Individual ring yield
    cluster_defect_rate: float = 0.02     # Clustered defects per cm²
    die_area_cm2: float = 0.25            # 5mm × 5mm die
    
    # Cost model
    wafer_cost_USD: int = 8000            # 150mm AlGaAsOI wafer
    dies_per_wafer: int = 50              # Conservative estimate
    packaging_cost_USD: int = 15          # InFO_oS packaging
    test_time_minutes: float = 2.0        # Calibration time per die
    
    def compute_die_yield(self) -> float:
        """Compute overall die yield including clustered defects."""
        # Individual ring yield
        ring_yield = self.ring_yield_probability
        
        # Clustered defect yield (Poisson model)
        cluster_yield = np.exp(-self.cluster_defect_rate * self.die_area_cm2)
        
        # Combined yield for minimum functional rings (3200 of 4000)
        min_functional = 3200
        total_rings = 4000
        
        # Binomial probability of having at least min_functional rings
        from scipy.stats import binom
        functional_yield = 1 - binom.cdf(min_functional - 1, total_rings, ring_yield)
        
        return functional_yield * cluster_yield
    
    def compute_unit_cost(self, volume: int = 1000) -> float:
        """Compute unit cost including yield and volume effects."""
        die_yield = self.compute_die_yield()
        
        # Die cost
        die_cost = self.wafer_cost_USD / (self.dies_per_wafer * die_yield)
        
        # Volume pricing (learning curve effect)
        volume_factor = max(0.5, 1000 / volume) ** 0.2  # 20% learning curve
        
        # Total unit cost
        total_cost = (die_cost + self.packaging_cost_USD) * volume_factor
        
        return total_cost


@dataclass
class ThermalModel:
    """Thermal modeling for the accelerator system."""
    
    # Die thermal properties
    die_size_mm: float = 5.0
    die_thickness_um: float = 200.0
    thermal_conductivity: float = 55.0  # AlGaAs W/m·K
    
    # Heat sources
    laser_power_W: float = 0.53
    ring_power_W: float = 0.20
    sram_power_W: float = 0.35
    
    # Thermal constraints
    max_temp_C: float = 85.0
    ambient_temp_C: float = 35.0  # Mobile ambient
    max_gradient_C: float = 10.0
    
    def __post_init__(self):
        """Initialize thermal resistance network."""
        # Simplified thermal resistance model
        # In production, this would import COMSOL data
        self.thermal_resistance = self._compute_thermal_resistance()
    
    def _compute_thermal_resistance(self) -> float:
        """Compute thermal resistance from die to ambient."""
        # Simplified model: R_th = thickness / (k * area)
        area_m2 = (self.die_size_mm * 1e-3) ** 2
        thickness_m = self.die_thickness_um * 1e-6
        
        return thickness_m / (self.thermal_conductivity * area_m2)
    
    def compute_die_temperature(self, total_power_W: float) -> float:
        """Compute peak die temperature."""
        return self.ambient_temp_C + total_power_W * self.thermal_resistance
    
    def compute_thermal_gradient(self, power_map: np.ndarray) -> float:
        """Compute thermal gradient across die (simplified)."""
        # In production, this would use detailed FEM analysis
        power_std = np.std(power_map)
        return power_std * self.thermal_resistance * 10  # Empirical factor
    
    def is_thermally_feasible(self, total_power_W: float, power_map: np.ndarray) -> bool:
        """Check if configuration meets thermal constraints."""
        peak_temp = self.compute_die_temperature(total_power_W)
        gradient = self.compute_thermal_gradient(power_map)
        
        return (peak_temp <= self.max_temp_C and 
                gradient <= self.max_gradient_C)


@dataclass
class SystemPerformanceModel:
    """Performance modeling for the full accelerator system."""
    
    # Architecture parameters
    num_mma_units: int = 68  # 4×4 tiles × 17 lanes
    rings_per_mma: int = 64  # 8×8 ring array per MMA
    clock_freq_GHz: float = 1.0
    
    # Memory system
    sram_bandwidth_TBps: float = 2.0
    sram_latency_ns: float = 5.0
    
    # Optical parameters
    modulation_bandwidth_GHz: float = 10.0
    optical_loss_budget_dB: float = 15.0
    
    def compute_peak_tops(self) -> float:
        """Compute peak TOPS performance."""
        # Each MMA performs matrix multiplication
        ops_per_mma = self.rings_per_mma * 2  # MAC operations
        total_ops = self.num_mma_units * ops_per_mma * self.clock_freq_GHz * 1e9
        
        return total_ops / 1e12  # Convert to TOPS
    
    def compute_sustained_tops(self, power_budget_W: float, thermal_model: ThermalModel) -> float:
        """Compute sustained TOPS under thermal constraints."""
        peak_tops = self.compute_peak_tops()
        
        # Thermal throttling factor
        if thermal_model.compute_die_temperature(power_budget_W) > thermal_model.max_temp_C:
            # Linear throttling model
            throttle_factor = thermal_model.max_temp_C / thermal_model.compute_die_temperature(power_budget_W)
        else:
            throttle_factor = 1.0
        
        return peak_tops * throttle_factor
    
    def compute_token_rate(self, model_size_B: int = 7, sustained_tops: float = None) -> float:
        """Compute token generation rate for LLM inference."""
        if sustained_tops is None:
            sustained_tops = self.compute_peak_tops()
        
        # Empirical model: tok/s ≈ TOPS / (model_size_B * complexity_factor)
        complexity_factor = 0.1  # Empirical for transformer models
        
        return sustained_tops / (model_size_B * complexity_factor)


@dataclass
class PhotonicAcceleratorOptimizer(ObjectiveFunction):
    """Level 4 system-level optimization for photonic AI accelerator."""
    
    dims: int = 25
    turn: float = 0.01
    name: str = "photonic_accelerator_l4"
    
    # System models
    config: AcceleratorConfig = field(default_factory=AcceleratorConfig)
    manufacturing: ManufacturingModel = field(default_factory=ManufacturingModel)
    thermal: ThermalModel = field(default_factory=ThermalModel)
    performance: SystemPerformanceModel = field(default_factory=SystemPerformanceModel)
    
    def __post_init__(self):
        """Initialize system-level optimization parameters."""
        # Extended parameter space for full accelerator system:
        # [0-2]: Ring array geometry (rows, cols, spacing_um)
        # [3-5]: Optical parameters (wavelength_nm, power_per_lane_mW, loss_budget_dB)
        # [6-8]: Thermal parameters (heater_power_uW, thermal_time_constant, gradient_limit)
        # [9-11]: Manufacturing (cd_target_nm, yield_target, process_corner)
        # [12-14]: System architecture (num_lanes, clock_freq_GHz, sram_config)
        # [15-17]: Power distribution (laser_power_W, ring_power_W, sram_power_W)
        # [18-20]: Performance targets (sustained_tops, token_rate, latency_ms)
        # [21-24]: Integration (package_type, test_time_min, calibration_points, compiler_config)
        
        self.lb = np.array([
            # Ring geometry (realistic bounds)
            32, 32, 8.0,
            # Optical (realistic wavelength and power)
            1530, 15, 12,
            # Thermal (realistic heater power and timing)
            30, 5e-6, 8.0,
            # Manufacturing (realistic CD and yield)
            210, 0.7, 0,
            # Architecture (realistic lanes and clock)
            12, 0.8, 256,
            # Power (mobile constraints)
            0.4, 0.15, 0.25,
            # Performance (realistic targets)
            2.5, 35, 8.0,
            # Integration (realistic test and calibration)
            0, 1.2, 12, 0
        ])
        
        self.ub = np.array([
            # Ring geometry
            96, 96, 15.0,
            # Optical
            1570, 35, 20,
            # Thermal
            80, 8e-6, 15.0,
            # Manufacturing
            280, 0.90, 2,
            # Architecture
            20, 1.5, 768,
            # Power (strict mobile limits)
            0.7, 0.25, 0.45,
            # Performance
            4.0, 65, 15.0,
            # Integration
            2, 2.5, 24, 2
        ])
        
        self.tracker = Tracker(self.name + str(self.dims))
        
        # Parameter names for debugging
        self.param_names = [
            "array_rows", "array_cols", "ring_spacing_um",
            "wavelength_nm", "power_per_lane_mW", "loss_budget_dB",
            "heater_power_uW", "thermal_tau_us", "gradient_limit_C",
            "cd_target_nm", "yield_target", "process_corner",
            "num_lanes", "clock_freq_GHz", "sram_config",
            "laser_power_W", "ring_power_W", "sram_power_W",
            "target_sustained_tops", "target_token_rate", "target_latency_ms",
            "package_type", "test_time_min", "calibration_points", "compiler_config"
        ]
    
    def scaled(self, y: float) -> float:
        """Scale composite score for DANTE optimization."""
        return max(0, y)
    
    def _extract_parameters(self, x: np.ndarray) -> Dict[str, Any]:
        """Extract and validate parameters from optimization vector."""
        params = {}
        
        # Ring array geometry
        params["array_rows"] = int(np.clip(x[0], 32, 128))
        params["array_cols"] = int(np.clip(x[1], 32, 128))
        params["ring_spacing_um"] = float(x[2])
        
        # Optical parameters
        params["wavelength_nm"] = float(x[3])
        params["power_per_lane_mW"] = float(x[4])
        params["loss_budget_dB"] = float(x[5])
        
        # Thermal parameters
        params["heater_power_uW"] = float(x[6])
        params["thermal_tau_us"] = float(x[7])
        params["gradient_limit_C"] = float(x[8])
        
        # Manufacturing parameters
        params["cd_target_nm"] = float(x[9])
        params["yield_target"] = float(x[10])
        params["process_corner"] = int(np.clip(x[11], 0, 2))  # SS/TT/FF
        
        # System architecture
        params["num_lanes"] = int(np.clip(x[12], 10, 25))
        params["clock_freq_GHz"] = float(x[13])
        params["sram_config"] = int(np.clip(x[14], 256, 1024))
        
        # Power distribution
        params["laser_power_W"] = float(x[15])
        params["ring_power_W"] = float(x[16])
        params["sram_power_W"] = float(x[17])
        
        # Performance targets
        params["target_sustained_tops"] = float(x[18])
        params["target_token_rate"] = float(x[19])
        params["target_latency_ms"] = float(x[20])
        
        # Integration parameters
        params["package_type"] = int(np.clip(x[21], 0, 2))
        params["test_time_min"] = float(x[22])
        params["calibration_points"] = int(np.clip(x[23], 8, 32))
        params["compiler_config"] = int(np.clip(x[24], 0, 2))
        
        return params
    
    def _compute_system_power(self, params: Dict[str, Any]) -> float:
        """Compute total system power consumption."""
        total_power = (
            params["laser_power_W"] +
            params["ring_power_W"] +
            params["sram_power_W"] +
            0.10 +  # 3R regeneration
            0.45 +  # ADC/DAC
            0.25    # Digital control
        )
        
        return total_power
    
    def _compute_manufacturing_yield(self, params: Dict[str, Any]) -> float:
        """Compute manufacturing yield with process variations."""
        # CD variation impact on ring resonance
        cd_variation = params["cd_target_nm"] * 0.02  # 2% variation
        wavelength_shift = cd_variation * 2.0  # Empirical sensitivity
        
        # Yield degradation from wavelength shift
        if wavelength_shift > 10.0:  # >10nm shift problematic
            yield_penalty = 0.5
        else:
            yield_penalty = 1.0 - wavelength_shift / 20.0
        
        # Process corner effects
        corner_factors = [0.8, 1.0, 0.9]  # SS, TT, FF
        corner_factor = corner_factors[params["process_corner"]]
        
        return params["yield_target"] * yield_penalty * corner_factor
    
    def _compute_thermal_performance(self, params: Dict[str, Any]) -> Tuple[float, bool]:
        """Compute thermal performance and feasibility."""
        total_power = self._compute_system_power(params)
        
        # Thermal model
        thermal_model = ThermalModel()
        peak_temp = thermal_model.compute_die_temperature(total_power)
        
        # Power distribution map (simplified)
        num_rings = params["array_rows"] * params["array_cols"]
        power_per_ring = params["ring_power_W"] / num_rings
        power_map = np.full((params["array_rows"], params["array_cols"]), power_per_ring)
        
        # Add laser hot spots
        power_map[0:4, 0:4] += params["laser_power_W"] / 16  # Laser region
        
        gradient = thermal_model.compute_thermal_gradient(power_map)
        thermal_feasible = thermal_model.is_thermally_feasible(total_power, power_map)
        
        return peak_temp, thermal_feasible
    
    def _compute_sustained_performance(self, params: Dict[str, Any]) -> float:
        """Compute sustained performance under realistic constraints."""
        # Base performance calculation
        performance_model = SystemPerformanceModel()
        performance_model.num_mma_units = params["num_lanes"] * 4  # 4 MMAs per lane
        performance_model.clock_freq_GHz = params["clock_freq_GHz"]
        
        peak_tops = performance_model.compute_peak_tops()
        
        # Thermal throttling
        total_power = self._compute_system_power(params)
        thermal_model = ThermalModel()
        sustained_tops = performance_model.compute_sustained_tops(total_power, thermal_model)
        
        # Yield impact on performance
        yield_factor = self._compute_manufacturing_yield(params)
        effective_tops = sustained_tops * yield_factor
        
        return effective_tops
    
    def _compute_cost_performance_ratio(self, params: Dict[str, Any]) -> float:
        """Compute cost-performance ratio ($/TOPS)."""
        sustained_tops = self._compute_sustained_performance(params)
        unit_cost = self.manufacturing.compute_unit_cost(volume=10000)  # 10K volume
        
        if sustained_tops <= 0:
            return 1e6  # Very high cost for invalid configs
        
        return unit_cost / sustained_tops
    
    def _compute_mobile_suitability_score(self, params: Dict[str, Any]) -> float:
        """Compute mobile suitability score (0-100)."""
        total_power = self._compute_system_power(params)
        peak_temp, thermal_feasible = self._compute_thermal_performance(params)
        sustained_tops = self._compute_sustained_performance(params)
        
        # Power score (0-100)
        if total_power <= 1.5:
            power_score = 100
        elif total_power >= 2.5:
            power_score = 0
        else:
            power_score = 100 * (2.5 - total_power) / 1.0
        
        # Thermal score (0-100)
        if thermal_feasible and peak_temp <= 75:
            thermal_score = 100
        elif not thermal_feasible:
            thermal_score = 0
        else:
            thermal_score = 100 * (85 - peak_temp) / 10
        
        # Performance score (0-100)
        if sustained_tops >= 3.0:
            perf_score = 100
        elif sustained_tops <= 1.0:
            perf_score = 0
        else:
            perf_score = 100 * (sustained_tops - 1.0) / 2.0
        
        # Weighted mobile score
        mobile_score = 0.4 * power_score + 0.3 * thermal_score + 0.3 * perf_score
        
        return mobile_score
    
    def _compute_manufacturing_feasibility_score(self, params: Dict[str, Any]) -> float:
        """Compute manufacturing feasibility score (0-100)."""
        yield_factor = self._compute_manufacturing_yield(params)
        unit_cost = self.manufacturing.compute_unit_cost(volume=10000)
        
        # Yield score
        yield_score = yield_factor * 100
        
        # Cost score (target <$50)
        if unit_cost <= 30:
            cost_score = 100
        elif unit_cost >= 100:
            cost_score = 0
        else:
            cost_score = 100 * (100 - unit_cost) / 70
        
        # Test time score (target <2 minutes)
        test_time = params["test_time_min"]
        if test_time <= 1.0:
            test_score = 100
        elif test_time >= 5.0:
            test_score = 0
        else:
            test_score = 100 * (5.0 - test_time) / 4.0
        
        # Weighted manufacturing score
        mfg_score = 0.5 * yield_score + 0.3 * cost_score + 0.2 * test_score
        
        return mfg_score
    
    def _run_system_simulation(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Run full system simulation with all constraints."""
        try:
            # System power analysis
            total_power = self._compute_system_power(params)
            
            # Thermal analysis
            peak_temp, thermal_feasible = self._compute_thermal_performance(params)
            
            # Performance analysis
            sustained_tops = self._compute_sustained_performance(params)
            token_rate = self.performance.compute_token_rate(sustained_tops=sustained_tops)
            
            # Manufacturing analysis
            yield_factor = self._compute_manufacturing_yield(params)
            unit_cost = self.manufacturing.compute_unit_cost(volume=10000)
            
            # Mobile suitability
            mobile_score = self._compute_mobile_suitability_score(params)
            mfg_score = self._compute_manufacturing_feasibility_score(params)
            
            return {
                "total_power_W": total_power,
                "peak_temp_C": peak_temp,
                "thermal_feasible": thermal_feasible,
                "sustained_tops": sustained_tops,
                "token_rate_per_s": token_rate,
                "yield_factor": yield_factor,
                "unit_cost_USD": unit_cost,
                "mobile_score": mobile_score,
                "manufacturing_score": mfg_score,
                "cost_per_tops": unit_cost / max(sustained_tops, 0.1),
                "power_efficiency_tops_per_w": sustained_tops / max(total_power, 0.1),
                "valid_config": (
                    total_power <= self.config.total_power_budget_W and
                    thermal_feasible and
                    sustained_tops >= 1.0 and
                    yield_factor >= 0.5
                )
            }
            
        except Exception as e:
            # Return penalty result for failed simulations
            return {
                "total_power_W": 10.0,
                "peak_temp_C": 150.0,
                "thermal_feasible": False,
                "sustained_tops": 0.0,
                "token_rate_per_s": 0.0,
                "yield_factor": 0.0,
                "unit_cost_USD": 1000.0,
                "mobile_score": 0.0,
                "manufacturing_score": 0.0,
                "cost_per_tops": 1e6,
                "power_efficiency_tops_per_w": 0.0,
                "valid_config": False
            }
    
    def __call__(self, x: np.ndarray, apply_scaling: bool = True, track: bool = True) -> float:
        """System-level optimization objective function."""
        x = self._preprocess(x)
        params = self._extract_parameters(x)
        result = self._run_system_simulation(params)
        
        # STRICT MOBILE CONSTRAINTS - Hard limits for mobile deployment
        total_power = result["total_power_W"]
        peak_temp = result["peak_temp_C"]
        
        # Hard constraint violations return 0 (invalid configuration)
        if total_power > self.config.total_power_budget_W:  # >2W invalid for mobile
            composite_score = 0.0
        elif peak_temp > 85.0:  # >85°C invalid for mobile
            composite_score = 0.0
        elif result["yield_factor"] < 0.5:  # <50% yield uneconomical
            composite_score = 0.0
        elif result["sustained_tops"] < 1.0:  # <1 TOPS insufficient performance
            composite_score = 0.0
        else:
            # Valid configuration - compute composite score
            power_efficiency = result["power_efficiency_tops_per_w"]
            mobile_suitability = result["mobile_score"]
            manufacturing_feasibility = result["manufacturing_score"]
            cost_effectiveness = 100 / max(result["cost_per_tops"], 1.0)
            
            # Weighted composite score for mobile AI accelerator
            composite_score = (
                0.35 * power_efficiency +      # Power efficiency critical for mobile
                0.25 * mobile_suitability +    # Mobile constraints (thermal, power)
                0.25 * manufacturing_feasibility + # Yield and cost
                0.15 * cost_effectiveness      # Economic viability
            )
        
        if track:
            self.tracker.track(composite_score, x)
            # Track detailed metrics for analysis
            try:
                self.tracker.track_metadata({
                    "total_power_W": result["total_power_W"],
                    "sustained_tops": result["sustained_tops"],
                    "token_rate": result["token_rate_per_s"],
                    "unit_cost": result["unit_cost_USD"],
                    "yield_factor": result["yield_factor"],
                    "thermal_feasible": result["thermal_feasible"],
                    "mobile_score": result["mobile_score"],
                    "mfg_score": result["manufacturing_score"],
                    "constraint_violations": {
                        "power_violation": total_power > self.config.total_power_budget_W,
                        "thermal_violation": peak_temp > 85.0,
                        "yield_violation": result["yield_factor"] < 0.5,
                        "performance_violation": result["sustained_tops"] < 1.0
                    }
                })
            except AttributeError:
                pass
        
        return composite_score if not apply_scaling else self.scaled(composite_score)


def optimize_photonic_accelerator(
    iterations: int = 500,
    initial_samples: int = 100,
    samples_per_acquisition: int = 20,
    target_power_W: float = 2.0,
    target_performance_tops: float = 3.11,
    output_file: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run Level 4 system optimization for the photonic AI accelerator.
    
    Args:
        iterations: Number of DANTE optimization iterations
        initial_samples: Number of initial random samples
        samples_per_acquisition: Samples per DANTE iteration
        target_power_W: Target power budget (mobile constraint)
        target_performance_tops: Target sustained performance
        output_file: Optional output file for results
        
    Returns:
        Optimization results with fab-ready accelerator configuration
    """
    # Import DANTE components
    from dante.neural_surrogate import AckleySurrogateModel
    from dante.tree_exploration import TreeExploration
    from dante.utils import generate_initial_samples
    
    # Create system-level objective function
    obj_function = PhotonicAcceleratorOptimizer()
    obj_function.config.total_power_budget_W = target_power_W
    obj_function.config.target_sustained_tops = target_performance_tops
    
    # Create surrogate model for system optimization
    surrogate = AckleySurrogateModel(input_dims=obj_function.dims, epochs=200)
    
    print(f"🚀 Starting Level 4 Photonic AI Accelerator Optimization...")
    print(f"Target: {target_performance_tops} TOPS at {target_power_W}W")
    print(f"Optimizing {obj_function.dims} system parameters...")
    
    # Generate initial samples
    input_x, input_y = generate_initial_samples(
        obj_function, num_init_samples=initial_samples, apply_scaling=True
    )
    
    best_solutions = []
    
    # Main optimization loop
    for i in range(iterations):
        # Train surrogate model
        trained_surrogate = surrogate(input_x, input_y)
        
        # Create tree explorer
        tree_explorer = TreeExploration(
            func=obj_function,
            model=trained_surrogate,
            num_samples_per_acquisition=samples_per_acquisition
        )
        
        # Perform tree exploration
        new_x = tree_explorer.rollout(input_x, input_y, iteration=i)
        new_y = np.array([obj_function(x, apply_scaling=True) for x in new_x])
        
        # Update dataset
        input_x = np.concatenate((input_x, new_x), axis=0)
        input_y = np.concatenate((input_y, new_y))
        
        # Track best solution
        best_idx = np.argmax(input_y)
        best_params = obj_function._extract_parameters(input_x[best_idx])
        best_result = obj_function._run_system_simulation(best_params)
        
        best_solutions.append({
            "iteration": i,
            "best_score": input_y[best_idx],
            "best_params": best_params,
            "system_metrics": best_result,
            "num_evaluations": len(input_y)
        })
        
        print(f"Iter {i+1}/{iterations}: Score={input_y[best_idx]:.2f}, "
              f"Power={best_result['total_power_W']:.2f}W, "
              f"TOPS={best_result['sustained_tops']:.2f}, "
              f"Cost=${best_result['unit_cost_USD']:.0f}")
        
        # Early stopping for convergence
        if i > 20 and len(set(input_y[-10:])) < 3:
            print(f"Converged after {i+1} iterations")
            break
    
    # Final results analysis
    best_idx = np.argmax(input_y)
    best_params = obj_function._extract_parameters(input_x[best_idx])
    best_result = obj_function._run_system_simulation(best_params)
    
    print(f"\n✅ Level 4 Optimization Complete!")
    print(f"Best Configuration:")
    print(f"  Power: {best_result['total_power_W']:.2f}W (target: {target_power_W}W)")
    print(f"  Performance: {best_result['sustained_tops']:.2f} TOPS (target: {target_performance_tops})")
    print(f"  Efficiency: {best_result['power_efficiency_tops_per_w']:.2f} TOPS/W")
    print(f"  Token Rate: {best_result['token_rate_per_s']:.1f} tok/s")
    print(f"  Unit Cost: ${best_result['unit_cost_USD']:.0f}")
    print(f"  Yield: {best_result['yield_factor']:.1%}")
    
    # Save results
    results = {
        "best_score": input_y[best_idx],
        "best_parameters": best_params,
        "best_metrics": best_result,
        "optimization_history": best_solutions,
        "all_evaluations": {"x": input_x.tolist(), "y": input_y.tolist()},
        "total_evaluations": len(input_y),
        "target_specs": {
            "power_budget_W": target_power_W,
            "performance_target_tops": target_performance_tops
        }
    }
    
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Results saved to {output_file}")
    
    return results


def generate_fab_ready_specs(optimization_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate fab-ready specifications from optimization results.
    
    Args:
        optimization_result: Results from optimize_photonic_accelerator()
        
    Returns:
        Fab-ready specifications for tape-out
    """
    best_params = optimization_result["best_parameters"]
    best_metrics = optimization_result["best_metrics"]
    
    # Generate comprehensive specifications
    fab_specs = {
        "design_specifications": {
            "die_size_mm": 5.0,
            "ring_array_config": f"{best_params['array_rows']}x{best_params['array_cols']}",
            "total_rings": best_params['array_rows'] * best_params['array_cols'],
            "ring_spacing_um": best_params["ring_spacing_um"],
            "operating_wavelength_nm": best_params["wavelength_nm"],
            "target_cd_nm": best_params["cd_target_nm"],
            "process_corner": ["SS", "TT", "FF"][best_params["process_corner"]]
        },
        
        "performance_specifications": {
            "sustained_tops": best_metrics["sustained_tops"],
            "power_efficiency_tops_per_w": best_metrics["power_efficiency_tops_per_w"],
            "token_rate_7B_model": best_metrics["token_rate_per_s"],
            "total_power_budget_W": best_metrics["total_power_W"],
            "target_latency_ms": best_params["target_latency_ms"]
        },
        
        "thermal_specifications": {
            "max_die_temp_C": best_metrics["peak_temp_C"],
            "thermal_feasible": best_metrics["thermal_feasible"],
            "heater_power_per_ring_uW": best_params["heater_power_uW"],
            "thermal_time_constant_us": best_params["thermal_tau_us"]
        },
        
        "manufacturing_specifications": {
            "target_yield": best_metrics["yield_factor"],
            "unit_cost_10k_volume": best_metrics["unit_cost_USD"],
            "cost_per_tops": best_metrics["cost_per_tops"],
            "test_time_minutes": best_params["test_time_min"],
            "calibration_points": best_params["calibration_points"]
        },
        
        "integration_specifications": {
            "package_type": ["InFO_oS", "CoWoS", "EMIB"][best_params["package_type"]],
            "sram_config_MB": best_params["sram_config"],
            "num_computational_lanes": best_params["num_lanes"],
            "clock_frequency_GHz": best_params["clock_freq_GHz"]
        },
        
        "compiler_targets": {
            "ring_array_size": [best_params['array_rows'], best_params['array_cols']],
            "wavelength_channels": 4,  # O-band channels
            "precision_bits": [2, 4, 8],  # Supported precisions
            "max_model_size_B": 13,  # Maximum model size
            "calibration_lut_points": best_params["calibration_points"]
        }
    }
    
    return fab_specs


def export_gds_parameters(fab_specs: Dict[str, Any], output_dir: str = "gds_export") -> None:
    """
    Export optimized parameters for GDS layout generation.
    
    Args:
        fab_specs: Fab-ready specifications from generate_fab_ready_specs()
        output_dir: Output directory for GDS parameters
    """
    Path(output_dir).mkdir(exist_ok=True)
    
    # Ring array layout parameters
    layout_params = {
        "ring_diameter_um": 10.0,  # Standard ring diameter
        "ring_spacing_um": fab_specs["design_specifications"]["ring_spacing_um"],
        "array_rows": fab_specs["design_specifications"]["ring_array_config"].split('x')[0],
        "array_cols": fab_specs["design_specifications"]["ring_array_config"].split('x')[1],
        "waveguide_width_nm": fab_specs["design_specifications"]["target_cd_nm"],
        "gap_width_nm": 200,  # Ring-waveguide gap
        "heater_width_um": 2.0,  # TiN heater width
        "contact_pitch_um": 5.0   # Electrical contact pitch
    }
    
    # Export for layout tools
    with open(f"{output_dir}/layout_parameters.json", 'w') as f:
        json.dump(layout_params, f, indent=2)
    
    # Export for process flow
    process_params = {
        "etch_depth_nm": 220,  # Full etch depth
        "sidewall_angle_deg": 85,  # Near-vertical sidewalls
        "surface_roughness_nm": 1.0,  # Target roughness
        "metal_stack": ["TiN", "Al", "SiO2"],  # Heater stack
        "passivation_thickness_nm": 500
    }
    
    with open(f"{output_dir}/process_parameters.json", 'w') as f:
        json.dump(process_params, f, indent=2)
    
    print(f"📐 GDS parameters exported to {output_dir}/")


def generate_test_patterns(fab_specs: Dict[str, Any], output_dir: str = "test_patterns") -> None:
    """
    Generate automated test patterns for production testing.
    
    Args:
        fab_specs: Fab-ready specifications
        output_dir: Output directory for test patterns
    """
    Path(output_dir).mkdir(exist_ok=True)
    
    # Calibration test patterns
    calibration_points = fab_specs["manufacturing_specifications"]["calibration_points"]
    wavelengths = np.linspace(1520, 1580, 4)  # O-band channels
    
    test_patterns = {
        "ring_calibration": {
            "wavelengths_nm": wavelengths.tolist(),
            "power_levels_mW": np.logspace(-2, 1, calibration_points).tolist(),
            "voltage_sweep_V": np.linspace(0, 5, 32).tolist(),
            "temperature_points_C": [25, 45, 65, 85]
        },
        
        "system_validation": {
            "test_vectors": [
                {"input_pattern": "all_zeros", "expected_output": "zero_vector"},
                {"input_pattern": "all_ones", "expected_output": "sum_vector"},
                {"input_pattern": "identity_matrix", "expected_output": "identity_response"},
                {"input_pattern": "random_sparse", "expected_output": "computed_result"}
            ],
            "performance_benchmarks": [
                {"model": "ResNet-18", "expected_tops": 2.5, "max_latency_ms": 8},
                {"model": "BERT-Base", "expected_tops": 3.0, "max_latency_ms": 12},
                {"model": "Llama-7B", "expected_tok_s": 50, "max_latency_ms": 20}
            ]
        }
    }
    
    with open(f"{output_dir}/test_patterns.json", 'w') as f:
        json.dump(test_patterns, f, indent=2)
    
    print(f"🧪 Test patterns exported to {output_dir}/")


def generate_compiler_config(fab_specs: Dict[str, Any], output_dir: str = "compiler_config") -> None:
    """
    Generate compiler configuration from optimized accelerator specs.
    
    Args:
        fab_specs: Fab-ready specifications
        output_dir: Output directory for compiler config
    """
    Path(output_dir).mkdir(exist_ok=True)
    
    compiler_config = {
        "target_architecture": {
            "name": "PhotonicAcceleratorV1",
            "ring_array_size": fab_specs["compiler_targets"]["ring_array_size"],
            "num_lanes": fab_specs["integration_specifications"]["num_computational_lanes"],
            "wavelength_channels": fab_specs["compiler_targets"]["wavelength_channels"],
            "clock_freq_MHz": fab_specs["integration_specifications"]["clock_frequency_GHz"] * 1000,
            "memory_bandwidth_GBps": 2000  # 2 TB/s
        },
        
        "precision_support": {
            "supported_bits": fab_specs["compiler_targets"]["precision_bits"],
            "default_precision": 8,
            "mixed_precision": True,
            "dynamic_quantization": True
        },
        
        "optimization_passes": {
            "thermal_aware_placement": True,
            "power_aware_scheduling": True,
            "yield_aware_mapping": True,
            "latency_optimization": True
        },
        
        "calibration_config": {
            "lut_points": fab_specs["compiler_targets"]["calibration_lut_points"],
            "wavelength_channels": 4,
            "temperature_compensation": True,
            "drift_prediction": True
        },
        
        "runtime_config": {
            "max_model_size_B": fab_specs["compiler_targets"]["max_model_size_B"],
            "batch_size_range": [1, 16],
            "sequence_length_range": [128, 2048],
            "memory_management": "dynamic"
        }
    }
    
    with open(f"{output_dir}/compiler_config.json", 'w') as f:
        json.dump(compiler_config, f, indent=2)
    
    print(f"⚙️ Compiler config exported to {output_dir}/")
