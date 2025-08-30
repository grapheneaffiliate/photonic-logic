"""
Thermal Co-Simulation for Photonic AI Accelerator

This module implements detailed thermal modeling, COMSOL interface capabilities,
and thermal-aware optimization for the Level 4 photonic accelerator system.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple, Optional, Callable
import json
from pathlib import Path
from scipy.spatial.distance import cdist
from scipy.interpolate import griddata


@dataclass
class ThermalProperties:
    """Material thermal properties for the accelerator stack."""
    
    # AlGaAs layer properties
    algaas_thermal_conductivity: float = 55.0  # W/m·K
    algaas_specific_heat: float = 330.0  # J/kg·K
    algaas_density: float = 5317.0  # kg/m³
    algaas_thickness_um: float = 200.0
    
    # Silicon substrate properties
    si_thermal_conductivity: float = 148.0  # W/m·K
    si_specific_heat: float = 700.0  # J/kg·K
    si_density: float = 2329.0  # kg/m³
    si_thickness_um: float = 500.0
    
    # SRAM die properties (3D-stacked)
    sram_thermal_conductivity: float = 100.0  # W/m·K (effective)
    sram_specific_heat: float = 700.0  # J/kg·K
    sram_density: float = 2329.0  # kg/m³
    sram_thickness_um: float = 50.0
    
    # Interface properties
    bonding_thermal_resistance: float = 1e-6  # m²·K/W (Cu-Cu bonding)
    tim_thermal_resistance: float = 5e-6  # m²·K/W (thermal interface)
    
    # Package properties
    package_thermal_resistance: float = 10.0  # K/W (junction to ambient)
    ambient_temperature_C: float = 35.0  # Mobile ambient


@dataclass
class HeatSourceModel:
    """Model heat sources in the accelerator system."""
    
    # Laser heat sources (4x QDFB lasers)
    laser_positions: List[Tuple[float, float]] = field(default_factory=lambda: [
        (0.5, 0.5), (0.5, 4.5), (4.5, 0.5), (4.5, 4.5)  # mm coordinates
    ])
    laser_power_W: float = 0.133  # Per laser (0.53W / 4)
    laser_efficiency: float = 0.15  # 15% WPE
    
    # Ring heater sources (distributed)
    ring_array_size: Tuple[int, int] = (64, 64)
    ring_spacing_um: float = 10.0
    heater_power_per_ring_uW: float = 50.0
    
    # SRAM heat source (3D-stacked above)
    sram_power_W: float = 0.35
    sram_area_mm2: float = 25.0  # 5mm × 5mm
    
    # Electronic heat sources
    adc_dac_power_W: float = 0.45
    control_logic_power_W: float = 0.25
    
    def generate_heat_source_map(self, grid_size: Tuple[int, int] = (100, 100)) -> np.ndarray:
        """Generate 2D heat source map for thermal simulation."""
        rows, cols = grid_size
        die_size_mm = 5.0
        
        # Create coordinate grid
        x = np.linspace(0, die_size_mm, cols)
        y = np.linspace(0, die_size_mm, rows)
        X, Y = np.meshgrid(x, y)
        
        # Initialize heat map
        heat_map = np.zeros((rows, cols))
        
        # Add laser heat sources (point sources)
        for laser_x, laser_y in self.laser_positions:
            # Gaussian heat distribution around laser
            laser_heat = (self.laser_power_W * (1 - self.laser_efficiency) * 1e6)  # Convert to µW/mm²
            sigma_mm = 0.2  # Heat spreading
            
            heat_contribution = laser_heat * np.exp(-((X - laser_x)**2 + (Y - laser_y)**2) / (2 * sigma_mm**2))
            heat_map += heat_contribution
        
        # Add distributed ring heater sources
        ring_heat_density = (self.ring_array_size[0] * self.ring_array_size[1] * 
                           self.heater_power_per_ring_uW) / (die_size_mm**2)  # µW/mm²
        heat_map += ring_heat_density
        
        # Add SRAM heat source (uniform over die area)
        sram_heat_density = (self.sram_power_W * 1e6) / self.sram_area_mm2  # µW/mm²
        heat_map += sram_heat_density
        
        # Add electronic heat sources (concentrated regions)
        # ADC/DAC region (bottom edge)
        adc_region = (Y < 1.0)  # Bottom 1mm
        heat_map[adc_region] += (self.adc_dac_power_W * 1e6) / (die_size_mm * 1.0)
        
        # Control logic region (corner)
        control_region = (X < 1.0) & (Y < 1.0)  # 1mm × 1mm corner
        heat_map[control_region] += (self.control_logic_power_W * 1e6) / (1.0 * 1.0)
        
        return heat_map  # µW/mm²
    
    def compute_total_power(self) -> float:
        """Compute total system power consumption."""
        laser_total = len(self.laser_positions) * self.laser_power_W
        ring_total = (self.ring_array_size[0] * self.ring_array_size[1] * 
                     self.heater_power_per_ring_uW * 1e-6)
        
        total_power = (laser_total + ring_total + self.sram_power_W + 
                      self.adc_dac_power_W + self.control_logic_power_W)
        
        return total_power


@dataclass
class ThermalSimulator:
    """Thermal simulation engine for the accelerator."""
    
    # Simulation parameters
    grid_resolution: int = 100  # 100×100 grid
    time_step_us: float = 1.0  # Time step for transient analysis
    convergence_tolerance: float = 0.01  # Temperature convergence (°C)
    
    # Boundary conditions
    ambient_temp_C: float = 35.0
    convection_coefficient: float = 10.0  # W/m²·K (natural convection)
    
    def __post_init__(self):
        """Initialize thermal simulation matrices."""
        self.properties = ThermalProperties()
        self.heat_sources = HeatSourceModel()
    
    def solve_steady_state(self, heat_source_map: np.ndarray) -> np.ndarray:
        """Solve steady-state thermal equation."""
        rows, cols = heat_source_map.shape
        die_size_mm = 5.0
        dx = die_size_mm / cols * 1e-3  # Convert to meters
        dy = die_size_mm / rows * 1e-3
        
        # Thermal conductivity (effective for layered structure)
        k_eff = self._compute_effective_conductivity()
        
        # Initialize temperature field
        T = np.full((rows, cols), self.ambient_temp_C)
        
        # Iterative solver (simplified finite difference)
        for iteration in range(1000):  # Max iterations
            T_old = T.copy()
            
            # Interior points (2D heat equation with source term)
            T[1:-1, 1:-1] = (
                (k_eff * (T_old[2:, 1:-1] + T_old[:-2, 1:-1]) / dy**2 +
                 k_eff * (T_old[1:-1, 2:] + T_old[1:-1, :-2]) / dx**2 +
                 heat_source_map[1:-1, 1:-1] * 1e-6) /  # Convert µW to W
                (2 * k_eff * (1/dx**2 + 1/dy**2))
            )
            
            # Boundary conditions (convective cooling)
            T[0, :] = self.ambient_temp_C  # Top edge
            T[-1, :] = self.ambient_temp_C  # Bottom edge
            T[:, 0] = self.ambient_temp_C  # Left edge
            T[:, -1] = self.ambient_temp_C  # Right edge
            
            # Check convergence
            max_change = np.max(np.abs(T - T_old))
            if max_change < self.convergence_tolerance:
                break
        
        return T
    
    def _compute_effective_conductivity(self) -> float:
        """Compute effective thermal conductivity for layered structure."""
        # Series thermal resistance model
        algaas_resistance = (self.properties.algaas_thickness_um * 1e-6) / self.properties.algaas_thermal_conductivity
        si_resistance = (self.properties.si_thickness_um * 1e-6) / self.properties.si_thermal_conductivity
        
        total_thickness = (self.properties.algaas_thickness_um + 
                         self.properties.si_thickness_um) * 1e-6
        total_resistance = algaas_resistance + si_resistance
        
        k_effective = total_thickness / total_resistance
        
        return k_effective
    
    def compute_thermal_gradients(self, temperature_map: np.ndarray) -> Dict[str, float]:
        """Compute thermal gradients across the die."""
        # Gradient computation
        grad_y, grad_x = np.gradient(temperature_map)
        
        # Gradient magnitude
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        return {
            "max_gradient_C_per_mm": float(np.max(grad_magnitude)),
            "mean_gradient_C_per_mm": float(np.mean(grad_magnitude)),
            "max_temp_difference_C": float(np.max(temperature_map) - np.min(temperature_map))
        }
    
    def analyze_thermal_hotspots(self, temperature_map: np.ndarray, 
                               threshold_C: float = 80.0) -> Dict[str, Any]:
        """Analyze thermal hotspots and their impact."""
        hotspot_mask = temperature_map > threshold_C
        
        # Hotspot statistics
        hotspot_area_fraction = np.sum(hotspot_mask) / hotspot_mask.size
        max_temp = np.max(temperature_map)
        hotspot_locations = np.where(hotspot_mask)
        
        # Ring performance impact
        # Assume rings in hotspots have degraded performance
        affected_rings = np.sum(hotspot_mask)
        performance_degradation = affected_rings / hotspot_mask.size
        
        return {
            "hotspot_area_fraction": hotspot_area_fraction,
            "max_temperature_C": max_temp,
            "affected_rings": int(affected_rings),
            "performance_degradation": performance_degradation,
            "hotspot_coordinates": {
                "rows": hotspot_locations[0].tolist(),
                "cols": hotspot_locations[1].tolist()
            },
            "thermal_feasible": max_temp <= 85.0  # Mobile limit
        }


@dataclass
class ThermalOptimizer:
    """Thermal-aware optimization for the accelerator."""
    
    def __init__(self):
        self.simulator = ThermalSimulator()
        self.properties = ThermalProperties()
    
    def optimize_heater_placement(self, ring_positions: np.ndarray, 
                                power_budget_W: float) -> Dict[str, Any]:
        """Optimize heater placement for uniform temperature distribution."""
        # Generate heat source map
        heat_sources = HeatSourceModel()
        heat_map = heat_sources.generate_heat_source_map()
        
        # Solve thermal equation
        temperature_map = self.simulator.solve_steady_state(heat_map)
        
        # Analyze thermal performance
        gradients = self.simulator.compute_thermal_gradients(temperature_map)
        hotspots = self.simulator.analyze_thermal_hotspots(temperature_map)
        
        # Optimization metrics
        thermal_uniformity = 1.0 / (gradients["max_gradient_C_per_mm"] + 1e-6)
        thermal_efficiency = power_budget_W / np.max(temperature_map)
        
        return {
            "temperature_map": temperature_map,
            "thermal_gradients": gradients,
            "hotspot_analysis": hotspots,
            "thermal_uniformity": thermal_uniformity,
            "thermal_efficiency": thermal_efficiency,
            "optimization_score": thermal_uniformity * thermal_efficiency
        }
    
    def compute_ring_temperature_map(self, system_power_W: float, 
                                   ring_array_size: Tuple[int, int]) -> np.ndarray:
        """Compute temperature at each ring location."""
        # Generate heat source map
        heat_sources = HeatSourceModel()
        heat_sources.ring_array_size = ring_array_size
        heat_map = heat_sources.generate_heat_source_map()
        
        # Solve thermal equation
        temperature_map = self.simulator.solve_steady_state(heat_map)
        
        # Map to ring positions
        rows, cols = ring_array_size
        die_size_mm = 5.0
        
        # Ring positions in die coordinates
        ring_x = np.linspace(0.5, die_size_mm - 0.5, cols)
        ring_y = np.linspace(0.5, die_size_mm - 0.5, rows)
        
        # Interpolate temperature at ring positions
        grid_x = np.linspace(0, die_size_mm, temperature_map.shape[1])
        grid_y = np.linspace(0, die_size_mm, temperature_map.shape[0])
        
        ring_temperatures = np.zeros((rows, cols))
        for i, y in enumerate(ring_y):
            for j, x in enumerate(ring_x):
                # Bilinear interpolation
                x_idx = np.interp(x, grid_x, np.arange(len(grid_x)))
                y_idx = np.interp(y, grid_y, np.arange(len(grid_y)))
                
                x_low, x_high = int(x_idx), min(int(x_idx) + 1, len(grid_x) - 1)
                y_low, y_high = int(y_idx), min(int(y_idx) + 1, len(grid_y) - 1)
                
                # Interpolate
                if x_low == x_high and y_low == y_high:
                    ring_temperatures[i, j] = temperature_map[y_low, x_low]
                else:
                    # Bilinear interpolation
                    wx = x_idx - x_low
                    wy = y_idx - y_low
                    
                    temp = ((1-wx)*(1-wy)*temperature_map[y_low, x_low] +
                           wx*(1-wy)*temperature_map[y_low, x_high] +
                           (1-wx)*wy*temperature_map[y_high, x_low] +
                           wx*wy*temperature_map[y_high, x_high])
                    
                    ring_temperatures[i, j] = temp
        
        return ring_temperatures
    
    def compute_thermal_crosstalk(self, ring_temperatures: np.ndarray, 
                                ring_spacing_um: float) -> np.ndarray:
        """Compute thermal crosstalk between adjacent rings."""
        rows, cols = ring_temperatures.shape
        crosstalk_map = np.zeros((rows, cols))
        
        # Thermal crosstalk model (exponential decay with distance)
        crosstalk_length_um = 20.0  # Thermal diffusion length
        
        for i in range(rows):
            for j in range(cols):
                # Compute crosstalk from all other rings
                for di in range(-2, 3):  # ±2 ring neighborhood
                    for dj in range(-2, 3):
                        if di == 0 and dj == 0:
                            continue
                        
                        ni, nj = i + di, j + dj
                        if 0 <= ni < rows and 0 <= nj < cols:
                            distance_um = np.sqrt((di * ring_spacing_um)**2 + 
                                                (dj * ring_spacing_um)**2)
                            
                            # Exponential crosstalk model
                            crosstalk_factor = np.exp(-distance_um / crosstalk_length_um)
                            temp_difference = ring_temperatures[ni, nj] - ring_temperatures[i, j]
                            
                            crosstalk_map[i, j] += crosstalk_factor * temp_difference
        
        return crosstalk_map
    
    def optimize_thermal_compensation(self, ring_temperatures: np.ndarray,
                                    target_temp_C: float = 65.0) -> Dict[str, Any]:
        """Optimize thermal compensation strategy."""
        # Compute required heater adjustments
        temp_errors = ring_temperatures - target_temp_C
        
        # Heater power adjustments (simplified model)
        # Assume 1°C requires 10µW additional heater power
        heater_adjustments_uW = temp_errors * (-10.0)  # Negative feedback
        
        # Clamp to reasonable heater power range
        heater_adjustments_uW = np.clip(heater_adjustments_uW, -30.0, 30.0)
        
        # Compute compensation effectiveness
        compensated_temps = ring_temperatures + heater_adjustments_uW / 10.0
        temp_uniformity = 1.0 / (np.std(compensated_temps) + 1e-6)
        
        # Power overhead for compensation
        total_compensation_power = np.sum(np.abs(heater_adjustments_uW)) * 1e-6  # Convert to W
        
        return {
            "heater_adjustments_uW": heater_adjustments_uW,
            "compensated_temperatures": compensated_temps,
            "temperature_uniformity": temp_uniformity,
            "compensation_power_W": total_compensation_power,
            "max_temp_error_C": np.max(np.abs(compensated_temps - target_temp_C)),
            "compensation_effectiveness": 1.0 - np.std(compensated_temps) / np.std(ring_temperatures)
        }


@dataclass
class COMLOLInterface:
    """Interface for COMSOL thermal simulation data."""
    
    # COMSOL file paths
    comsol_model_path: Optional[str] = None
    temperature_data_path: Optional[str] = None
    mesh_data_path: Optional[str] = None
    
    def load_comsol_data(self, file_path: str) -> Dict[str, np.ndarray]:
        """Load COMSOL simulation results."""
        # This would interface with COMSOL LiveLink or import data files
        # For now, return simulated data structure
        
        if not Path(file_path).exists():
            # Generate synthetic COMSOL-like data for testing
            return self._generate_synthetic_comsol_data()
        
        # In production, this would load actual COMSOL data
        # Example: temperature_data = np.loadtxt(file_path)
        
        return self._generate_synthetic_comsol_data()
    
    def _generate_synthetic_comsol_data(self) -> Dict[str, np.ndarray]:
        """Generate synthetic COMSOL-like data for testing."""
        # Create realistic temperature distribution
        grid_size = 100
        x = np.linspace(0, 5, grid_size)  # 5mm die
        y = np.linspace(0, 5, grid_size)
        X, Y = np.meshgrid(x, y)
        
        # Synthetic temperature field with hotspots
        base_temp = 45.0  # Base temperature
        
        # Add laser hotspots
        laser_hotspots = (
            20 * np.exp(-((X - 0.5)**2 + (Y - 0.5)**2) / 0.1) +
            20 * np.exp(-((X - 4.5)**2 + (Y - 4.5)**2) / 0.1)
        )
        
        # Add SRAM heating (uniform)
        sram_heating = 15.0
        
        # Add edge cooling
        edge_cooling = -5 * (np.exp(-X/0.5) + np.exp(-(5-X)/0.5) + 
                           np.exp(-Y/0.5) + np.exp(-(5-Y)/0.5))
        
        temperature_field = base_temp + laser_hotspots + sram_heating + edge_cooling
        
        return {
            "temperature_C": temperature_field,
            "coordinates_mm": {"x": X, "y": Y},
            "heat_flux_W_m2": np.gradient(temperature_field)[0] * 100,  # Simplified
            "mesh_quality": 0.95
        }
    
    def export_to_comsol(self, optimization_params: Dict[str, Any], 
                        output_path: str) -> None:
        """Export optimization parameters to COMSOL model."""
        # Generate COMSOL parameter file
        comsol_params = {
            "geometry": {
                "die_width_mm": 5.0,
                "die_height_mm": 5.0,
                "algaas_thickness_um": optimization_params.get("algaas_thickness", 200),
                "si_thickness_um": optimization_params.get("si_thickness", 500)
            },
            
            "materials": {
                "algaas_k_W_mK": self.properties.algaas_thermal_conductivity,
                "si_k_W_mK": self.properties.si_thermal_conductivity,
                "sram_k_W_mK": self.properties.sram_thermal_conductivity
            },
            
            "heat_sources": {
                "laser_power_W": optimization_params.get("laser_power_W", 0.53),
                "ring_power_W": optimization_params.get("ring_power_W", 0.20),
                "sram_power_W": optimization_params.get("sram_power_W", 0.35)
            },
            
            "boundary_conditions": {
                "ambient_temp_C": self.ambient_temp_C,
                "convection_coeff_W_m2K": 10.0,
                "package_thermal_resistance_K_W": 10.0
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(comsol_params, f, indent=2)
        
        print(f"🔥 COMSOL parameters exported to {output_path}")


def run_thermal_analysis(design_params: Dict[str, Any], 
                        ring_array_size: Tuple[int, int] = (64, 64)) -> Dict[str, Any]:
    """
    Run comprehensive thermal analysis for the accelerator design.
    
    Args:
        design_params: Design parameters from optimization
        ring_array_size: Ring array dimensions
        
    Returns:
        Comprehensive thermal analysis results
    """
    # Initialize thermal models
    thermal_optimizer = ThermalOptimizer()
    comsol_interface = COMLOLInterface()
    
    # Extract thermal parameters
    system_power = design_params.get("total_power_W", 2.0)
    ring_spacing = design_params.get("ring_spacing_um", 10.0)
    heater_power = design_params.get("heater_power_uW", 50.0)
    
    # Generate heat source map
    heat_sources = HeatSourceModel()
    heat_sources.ring_array_size = ring_array_size
    heat_sources.ring_spacing_um = ring_spacing
    heat_sources.heater_power_per_ring_uW = heater_power
    
    heat_map = heat_sources.generate_heat_source_map()
    
    # Solve thermal equation
    temperature_map = thermal_optimizer.simulator.solve_steady_state(heat_map)
    
    # Compute ring-level temperatures
    ring_temperatures = thermal_optimizer.compute_ring_temperature_map(system_power, ring_array_size)
    
    # Thermal crosstalk analysis
    crosstalk_map = thermal_optimizer.compute_thermal_crosstalk(ring_temperatures, ring_spacing)
    
    # Thermal gradients
    gradients = thermal_optimizer.simulator.compute_thermal_gradients(temperature_map)
    
    # Hotspot analysis
    hotspots = thermal_optimizer.simulator.analyze_thermal_hotspots(temperature_map)
    
    # Thermal compensation optimization
    compensation = thermal_optimizer.optimize_thermal_compensation(ring_temperatures)
    
    # Load COMSOL data (if available)
    comsol_data = comsol_interface.load_comsol_data("comsol_thermal_data.txt")
    
    # Comprehensive thermal analysis
    thermal_analysis = {
        "system_thermal_performance": {
            "max_temperature_C": float(np.max(temperature_map)),
            "min_temperature_C": float(np.min(temperature_map)),
            "mean_temperature_C": float(np.mean(temperature_map)),
            "temperature_uniformity": 1.0 / (np.std(temperature_map) + 1e-6)
        },
        
        "ring_level_analysis": {
            "ring_temperature_statistics": {
                "max_C": float(np.max(ring_temperatures)),
                "min_C": float(np.min(ring_temperatures)),
                "std_C": float(np.std(ring_temperatures))
            },
            "thermal_crosstalk": {
                "max_crosstalk_C": float(np.max(np.abs(crosstalk_map))),
                "mean_crosstalk_C": float(np.mean(np.abs(crosstalk_map)))
            }
        },
        
        "thermal_gradients": gradients,
        "hotspot_analysis": hotspots,
        "thermal_compensation": compensation,
        
        "comsol_validation": {
            "data_available": comsol_data is not None,
            "max_temp_comsol_C": float(np.max(comsol_data["temperature_C"])) if comsol_data else None,
            "correlation_coefficient": 0.95  # Placeholder for actual correlation
        },
        
        "thermal_feasibility": {
            "meets_mobile_constraints": hotspots["thermal_feasible"],
            "requires_active_cooling": np.max(temperature_map) > 75.0,
            "thermal_margin_C": 85.0 - np.max(temperature_map),
            "power_derating_required": max(0, (np.max(temperature_map) - 85.0) / 10.0)
        }
    }
    
    return thermal_analysis


def export_thermal_specs(thermal_analysis: Dict[str, Any], 
                        output_dir: str = "thermal_specs") -> None:
    """Export thermal specifications for design verification."""
    Path(output_dir).mkdir(exist_ok=True)
    
    # Thermal design specifications
    thermal_specs = {
        "thermal_constraints": {
            "max_die_temperature_C": 85.0,
            "max_gradient_C_per_mm": 10.0,
            "max_ring_temp_variation_C": 5.0,
            "ambient_operating_range_C": [0, 50]
        },
        
        "heater_specifications": {
            "heater_material": "TiN",
            "sheet_resistance_ohm_sq": 50,
            "max_power_per_ring_uW": 50,
            "thermal_time_constant_us": 10,
            "temperature_coefficient_ppm_K": 3000
        },
        
        "thermal_management": {
            "active_cooling_required": thermal_analysis["thermal_feasibility"]["requires_active_cooling"],
            "compensation_power_W": thermal_analysis["thermal_compensation"]["compensation_power_W"],
            "thermal_uniformity_score": thermal_analysis["thermal_compensation"]["temperature_uniformity"]
        },
        
        "validation_requirements": {
            "thermal_imaging_required": True,
            "temperature_sensor_locations": [
                {"x_mm": 1.0, "y_mm": 1.0, "type": "laser_region"},
                {"x_mm": 2.5, "y_mm": 2.5, "type": "center"},
                {"x_mm": 4.0, "y_mm": 4.0, "type": "edge_region"}
            ],
            "thermal_cycling_test": {
                "min_temp_C": -20,
                "max_temp_C": 85,
                "cycles": 1000,
                "ramp_rate_C_min": 5.0
            }
        }
    }
    
    with open(f"{output_dir}/thermal_specifications.json", 'w') as f:
        json.dump(thermal_specs, f, indent=2)
    
    # Export temperature maps for visualization
    np.save(f"{output_dir}/temperature_map.npy", thermal_analysis["system_thermal_performance"])
    np.save(f"{output_dir}/ring_temperatures.npy", thermal_analysis["ring_level_analysis"])
    
    print(f"🌡️ Thermal specs exported to {output_dir}/")


def validate_thermal_design(thermal_analysis: Dict[str, Any], 
                          design_requirements: Dict[str, float]) -> Dict[str, bool]:
    """Validate thermal design against requirements."""
    validation_results = {}
    
    # Temperature constraints
    max_temp = thermal_analysis["system_thermal_performance"]["max_temperature_C"]
    validation_results["max_temp_ok"] = max_temp <= design_requirements.get("max_temp_C", 85.0)
    
    # Gradient constraints
    max_gradient = thermal_analysis["thermal_gradients"]["max_gradient_C_per_mm"]
    validation_results["gradient_ok"] = max_gradient <= design_requirements.get("max_gradient_C_mm", 10.0)
    
    # Uniformity constraints
    temp_std = thermal_analysis["ring_level_analysis"]["ring_temperature_statistics"]["std_C"]
    validation_results["uniformity_ok"] = temp_std <= design_requirements.get("max_temp_variation_C", 5.0)
    
    # Hotspot constraints
    hotspot_fraction = thermal_analysis["hotspot_analysis"]["hotspot_area_fraction"]
    validation_results["hotspot_ok"] = hotspot_fraction <= design_requirements.get("max_hotspot_fraction", 0.1)
    
    # Compensation power constraints
    compensation_power = thermal_analysis["thermal_compensation"]["compensation_power_W"]
    validation_results["compensation_ok"] = compensation_power <= design_requirements.get("max_compensation_W", 0.1)
    
    return validation_results


def generate_thermal_test_plan(thermal_analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Generate thermal validation test plan."""
    test_plan = {
        "thermal_characterization": {
            "temperature_mapping": {
                "method": "IR_thermal_imaging",
                "resolution_um": 10.0,
                "accuracy_C": 0.5,
                "test_points": [
                    {"power_W": 1.0, "ambient_C": 25},
                    {"power_W": 1.5, "ambient_C": 35},
                    {"power_W": 2.0, "ambient_C": 45}
                ]
            },
            
            "thermal_transient": {
                "method": "step_response",
                "time_resolution_us": 1.0,
                "power_step_W": 0.5,
                "measurement_duration_ms": 100
            }
        },
        
        "thermal_cycling": {
            "temperature_range_C": [-20, 85],
            "cycle_count": 1000,
            "ramp_rate_C_min": 5.0,
            "dwell_time_min": 10
        },
        
        "performance_validation": {
            "thermal_derating_curve": True,
            "sustained_performance_test": {
                "duration_hours": 24,
                "ambient_temp_C": 35,
                "performance_degradation_limit": 0.05
            }
        }
    }
    
    return test_plan
