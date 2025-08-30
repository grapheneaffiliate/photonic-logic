"""
Manufacturing-Aware Constraints for Photonic AI Accelerator

This module implements detailed manufacturing constraints, process variation modeling,
and yield optimization for the Level 4 photonic accelerator system.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple, Optional
from scipy.stats import norm, binom
import json


@dataclass
class ProcessVariationModel:
    """Model process variations for AlGaAsOI fabrication."""
    
    # Critical dimension variations
    cd_mean_nm: float = 220.0
    cd_std_nm: float = 2.0  # 3σ = ±6nm
    cd_correlation_length_um: float = 100.0  # Spatial correlation
    
    # Thickness variations
    thickness_mean_nm: float = 200.0
    thickness_std_nm: float = 5.0
    
    # Refractive index variations
    n_eff_mean: float = 3.4
    n_eff_std: float = 0.02
    
    # Sidewall roughness
    sidewall_rms_nm: float = 1.5
    correlation_length_nm: float = 50.0
    
    def generate_cd_map(self, array_size: Tuple[int, int], spacing_um: float) -> np.ndarray:
        """Generate spatially correlated CD variation map."""
        rows, cols = array_size
        
        # Generate correlated random field
        x = np.arange(cols) * spacing_um
        y = np.arange(rows) * spacing_um
        X, Y = np.meshgrid(x, y)
        
        # Exponential correlation function
        correlation_matrix = np.exp(-np.sqrt((X[:, :, None, None] - X[None, None, :, :]) ** 2 + 
                                           (Y[:, :, None, None] - Y[None, None, :, :]) ** 2) / 
                                   self.cd_correlation_length_um)
        
        # Generate correlated variations
        random_field = np.random.randn(rows, cols)
        cd_variations = self.cd_std_nm * random_field
        
        # Apply spatial correlation (simplified)
        from scipy.ndimage import gaussian_filter
        sigma = self.cd_correlation_length_um / spacing_um / 3  # 3σ rule
        cd_variations = gaussian_filter(cd_variations, sigma=sigma)
        
        return self.cd_mean_nm + cd_variations
    
    def compute_wavelength_shift(self, cd_map: np.ndarray) -> np.ndarray:
        """Compute wavelength shift from CD variations."""
        # Empirical sensitivity: Δλ/λ ≈ Δw/w for ring resonators
        cd_fractional_change = (cd_map - self.cd_mean_nm) / self.cd_mean_nm
        wavelength_shift_nm = 1550 * cd_fractional_change  # Operating at 1550nm
        
        return wavelength_shift_nm
    
    def compute_q_factor_variation(self, cd_map: np.ndarray) -> np.ndarray:
        """Compute Q-factor variations from sidewall roughness and CD."""
        # Sidewall scattering loss model
        # α_scattering ∝ (σ_rms / w)² where σ_rms is sidewall roughness, w is width
        scattering_loss = (self.sidewall_rms_nm / cd_map) ** 2
        
        # Base Q-factor (intrinsic)
        q_intrinsic = 1e6
        
        # Q degradation from scattering
        q_factor = q_intrinsic / (1 + scattering_loss * 1000)  # Empirical factor
        
        return q_factor


@dataclass
class YieldModel:
    """Comprehensive yield modeling for the accelerator."""
    
    # Defect models
    random_defect_density: float = 0.1  # defects/cm²
    cluster_defect_density: float = 0.02  # clusters/cm²
    cluster_size_rings: int = 10  # Average rings per cluster
    
    # Functional requirements
    min_functional_rings: int = 3200  # Minimum for operation
    total_rings: int = 4000
    min_q_factor: float = 50000  # Minimum usable Q
    max_wavelength_error_nm: float = 2.0  # Tuning range limit
    
    # Electrical yield
    contact_yield: float = 0.98  # Per contact
    heater_yield: float = 0.95   # Per heater
    
    def compute_optical_yield(self, cd_map: np.ndarray, q_map: np.ndarray, 
                            wavelength_shift_map: np.ndarray) -> float:
        """Compute optical yield from device parameter maps."""
        # Q-factor yield
        q_good = np.sum(q_map >= self.min_q_factor)
        
        # Wavelength tuning yield
        wavelength_good = np.sum(np.abs(wavelength_shift_map) <= self.max_wavelength_error_nm)
        
        # Combined optical yield
        optical_good = np.sum((q_map >= self.min_q_factor) & 
                            (np.abs(wavelength_shift_map) <= self.max_wavelength_error_nm))
        
        return optical_good / self.total_rings
    
    def compute_electrical_yield(self, array_size: Tuple[int, int]) -> float:
        """Compute electrical yield for contacts and heaters."""
        total_elements = array_size[0] * array_size[1]
        
        # Independent failures for contacts and heaters
        contact_yield_total = self.contact_yield ** total_elements
        heater_yield_total = self.heater_yield ** total_elements
        
        return contact_yield_total * heater_yield_total
    
    def compute_clustered_defect_yield(self, die_area_cm2: float) -> float:
        """Compute yield from clustered defects (killer defects)."""
        # Poisson model for cluster occurrence
        expected_clusters = self.cluster_defect_density * die_area_cm2
        cluster_yield = np.exp(-expected_clusters)
        
        return cluster_yield
    
    def compute_overall_yield(self, cd_map: np.ndarray, q_map: np.ndarray,
                            wavelength_shift_map: np.ndarray, die_area_cm2: float,
                            array_size: Tuple[int, int]) -> Dict[str, float]:
        """Compute overall die yield with detailed breakdown."""
        # Individual yield components
        optical_yield = self.compute_optical_yield(cd_map, q_map, wavelength_shift_map)
        electrical_yield = self.compute_electrical_yield(array_size)
        cluster_yield = self.compute_clustered_defect_yield(die_area_cm2)
        
        # Functional yield (minimum rings requirement)
        optical_good_rings = np.sum((q_map >= self.min_q_factor) & 
                                  (np.abs(wavelength_shift_map) <= self.max_wavelength_error_nm))
        
        # Binomial probability of meeting minimum requirement
        functional_yield = 1 - binom.cdf(self.min_functional_rings - 1, 
                                       self.total_rings, optical_yield)
        
        # Overall yield
        overall_yield = functional_yield * electrical_yield * cluster_yield
        
        return {
            "optical_yield": optical_yield,
            "electrical_yield": electrical_yield,
            "cluster_yield": cluster_yield,
            "functional_yield": functional_yield,
            "overall_yield": overall_yield,
            "functional_rings": optical_good_rings,
            "yield_breakdown": {
                "q_factor_limited": np.sum(q_map < self.min_q_factor) / self.total_rings,
                "wavelength_limited": np.sum(np.abs(wavelength_shift_map) > self.max_wavelength_error_nm) / self.total_rings,
                "electrical_limited": 1 - electrical_yield,
                "cluster_limited": 1 - cluster_yield
            }
        }


@dataclass
class ProcessCornerModel:
    """Model process corners (SS/TT/FF) for the accelerator."""
    
    # Process corner definitions
    corners: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "SS": {  # Slow-Slow (worst case)
            "cd_bias_nm": +3.0,
            "thickness_bias_nm": +10.0,
            "n_eff_bias": +0.01,
            "loss_multiplier": 1.5,
            "thermal_resistance_multiplier": 1.2
        },
        "TT": {  # Typical-Typical (nominal)
            "cd_bias_nm": 0.0,
            "thickness_bias_nm": 0.0,
            "n_eff_bias": 0.0,
            "loss_multiplier": 1.0,
            "thermal_resistance_multiplier": 1.0
        },
        "FF": {  # Fast-Fast (best case)
            "cd_bias_nm": -3.0,
            "thickness_bias_nm": -10.0,
            "n_eff_bias": -0.01,
            "loss_multiplier": 0.8,
            "thermal_resistance_multiplier": 0.9
        }
    })
    
    def apply_corner_effects(self, corner: str, base_params: Dict[str, Any]) -> Dict[str, Any]:
        """Apply process corner effects to base parameters."""
        if corner not in self.corners:
            raise ValueError(f"Unknown process corner: {corner}")
        
        corner_data = self.corners[corner]
        modified_params = base_params.copy()
        
        # Apply corner biases
        modified_params["cd_target_nm"] += corner_data["cd_bias_nm"]
        modified_params["thickness_nm"] = base_params.get("thickness_nm", 200) + corner_data["thickness_bias_nm"]
        modified_params["n_eff"] = base_params.get("n_eff", 3.4) + corner_data["n_eff_bias"]
        
        # Apply multipliers
        modified_params["loss_multiplier"] = corner_data["loss_multiplier"]
        modified_params["thermal_multiplier"] = corner_data["thermal_resistance_multiplier"]
        
        return modified_params
    
    def compute_corner_yield(self, base_yield: float, corner: str) -> float:
        """Compute yield for specific process corner."""
        corner_factors = {
            "SS": 0.7,  # Worst case yield
            "TT": 1.0,  # Nominal yield
            "FF": 0.9   # Best case (but may have other issues)
        }
        
        return base_yield * corner_factors.get(corner, 1.0)


@dataclass
class ReliabilityModel:
    """Long-term reliability modeling for mobile deployment."""
    
    # Operating conditions
    operating_temp_C: float = 65.0  # Average mobile temperature
    humidity_percent: float = 60.0
    thermal_cycles_per_day: int = 50  # Mobile usage pattern
    
    # Failure mechanisms
    electromigration_activation_eV: float = 0.7  # TiN heaters
    thermal_fatigue_cycles: int = 1e6  # Solder joint fatigue
    optical_degradation_rate: float = 1e-9  # Q-factor degradation per hour
    
    # Reliability targets
    target_lifetime_years: float = 5.0
    target_failure_rate: float = 0.01  # 1% failure rate
    
    def compute_electromigration_mtf(self, current_density_A_cm2: float, 
                                   temperature_C: float) -> float:
        """Compute mean time to failure from electromigration."""
        # Black's equation: MTF = A * J^(-n) * exp(Ea/kT)
        A = 1e10  # Material constant
        n = 2.0   # Current density exponent
        k_eV = 8.617e-5  # Boltzmann constant in eV/K
        
        mtf_hours = A * (current_density_A_cm2 ** (-n)) * np.exp(
            self.electromigration_activation_eV / (k_eV * (temperature_C + 273.15))
        )
        
        return mtf_hours
    
    def compute_thermal_fatigue_cycles(self, delta_temp_C: float) -> int:
        """Compute thermal fatigue cycles to failure."""
        # Coffin-Manson model: N = A * (ΔT)^(-n)
        A = 1e8
        n = 2.0
        
        cycles_to_failure = A * (delta_temp_C ** (-n))
        
        return int(cycles_to_failure)
    
    def compute_optical_degradation(self, operating_hours: float) -> float:
        """Compute optical performance degradation over time."""
        # Exponential degradation model
        degradation_factor = np.exp(-self.optical_degradation_rate * operating_hours)
        
        return degradation_factor
    
    def assess_reliability(self, heater_current_mA: float, max_temp_C: float,
                         thermal_swing_C: float) -> Dict[str, Any]:
        """Assess overall reliability for mobile deployment."""
        # Convert current to current density (assuming 2μm × 100μm heater)
        heater_area_cm2 = 2e-4 * 100e-4  # cm²
        current_density = heater_current_mA * 1e-3 / heater_area_cm2
        
        # Compute failure mechanisms
        em_mtf_hours = self.compute_electromigration_mtf(current_density, max_temp_C)
        thermal_cycles = self.compute_thermal_fatigue_cycles(thermal_swing_C)
        
        # Operating hours over lifetime
        lifetime_hours = self.target_lifetime_years * 365 * 24
        
        # Reliability assessment
        em_reliability = np.exp(-lifetime_hours / em_mtf_hours)
        thermal_reliability = np.exp(-lifetime_hours * self.thermal_cycles_per_day / thermal_cycles)
        optical_degradation = self.compute_optical_degradation(lifetime_hours)
        
        # Combined reliability
        overall_reliability = em_reliability * thermal_reliability * optical_degradation
        
        return {
            "electromigration_mtf_years": em_mtf_hours / (365 * 24),
            "thermal_fatigue_cycles": thermal_cycles,
            "optical_degradation_factor": optical_degradation,
            "overall_reliability": overall_reliability,
            "meets_target": overall_reliability >= (1 - self.target_failure_rate),
            "limiting_mechanism": self._identify_limiting_mechanism(
                em_reliability, thermal_reliability, optical_degradation
            )
        }
    
    def _identify_limiting_mechanism(self, em_rel: float, thermal_rel: float, 
                                   optical_rel: float) -> str:
        """Identify the limiting reliability mechanism."""
        mechanisms = {
            "electromigration": em_rel,
            "thermal_fatigue": thermal_rel,
            "optical_degradation": optical_rel
        }
        
        return min(mechanisms, key=mechanisms.get)


@dataclass
class FoundryConstraints:
    """Foundry-specific manufacturing constraints."""
    
    # AlGaAsOI foundry capabilities
    foundry_name: str = "CompoundTek"
    wafer_size_mm: int = 150
    min_feature_size_nm: float = 180.0
    max_aspect_ratio: float = 10.0
    
    # Etch capabilities
    etch_depth_range_nm: Tuple[float, float] = (100.0, 400.0)
    etch_selectivity: float = 50.0  # AlGaAs:AlAs selectivity
    sidewall_angle_deg: float = 85.0
    
    # Metal stack constraints
    available_metals: List[str] = field(default_factory=lambda: ["TiN", "Al", "Au", "Pt"])
    max_metal_layers: int = 3
    via_size_range_um: Tuple[float, float] = (0.5, 5.0)
    
    # Thermal budget
    max_process_temp_C: float = 400.0  # AlGaAs thermal budget
    annealing_temp_C: float = 350.0
    
    # Design rules
    min_spacing_nm: float = 200.0
    min_via_enclosure_nm: float = 100.0
    max_metal_width_um: float = 100.0
    
    def validate_design_rules(self, design_params: Dict[str, Any]) -> Dict[str, bool]:
        """Validate design against foundry constraints."""
        violations = {}
        
        # Feature size check
        cd_nm = design_params.get("cd_target_nm", 220)
        violations["min_feature_size"] = cd_nm >= self.min_feature_size_nm
        
        # Aspect ratio check
        etch_depth = design_params.get("etch_depth_nm", 220)
        aspect_ratio = etch_depth / cd_nm
        violations["aspect_ratio"] = aspect_ratio <= self.max_aspect_ratio
        
        # Spacing check
        ring_spacing = design_params.get("ring_spacing_um", 10) * 1000  # Convert to nm
        violations["min_spacing"] = ring_spacing >= self.min_spacing_nm
        
        # Thermal budget check
        process_temp = design_params.get("process_temp_C", 300)
        violations["thermal_budget"] = process_temp <= self.max_process_temp_C
        
        # Metal stack check
        num_metal_layers = design_params.get("num_metal_layers", 2)
        violations["metal_stack"] = num_metal_layers <= self.max_metal_layers
        
        return violations
    
    def compute_design_rule_score(self, design_params: Dict[str, Any]) -> float:
        """Compute design rule compliance score (0-100)."""
        violations = self.validate_design_rules(design_params)
        
        # Count violations
        num_violations = sum(1 for v in violations.values() if not v)
        total_rules = len(violations)
        
        # Score based on compliance
        compliance_score = 100 * (total_rules - num_violations) / total_rules
        
        return compliance_score


@dataclass
class TestabilityModel:
    """Production test and calibration modeling."""
    
    # Test equipment constraints
    probe_card_cost_USD: int = 100000  # High-cost optical probes
    test_station_cost_USD: int = 500000
    throughput_dies_per_hour: int = 30  # Limited by calibration time
    
    # Calibration requirements
    calibration_points_per_ring: int = 16  # LUT size
    wavelength_channels: int = 4  # O-band channels
    temperature_points: int = 4  # 25, 45, 65, 85°C
    
    # Test time breakdown
    optical_alignment_s: float = 10.0
    electrical_test_s: float = 5.0
    calibration_per_ring_ms: float = 2.0  # Per ring calibration
    
    def compute_test_time(self, num_rings: int, calibration_points: int) -> float:
        """Compute total test time per die."""
        # Base test time
        base_time = self.optical_alignment_s + self.electrical_test_s
        
        # Calibration time
        calibration_time = (num_rings * calibration_points * 
                          self.calibration_per_ring_ms * 1e-3)
        
        total_time_minutes = (base_time + calibration_time) / 60
        
        return total_time_minutes
    
    def compute_test_cost_per_die(self, test_time_minutes: float) -> float:
        """Compute test cost per die."""
        # Amortized equipment cost
        equipment_cost_per_hour = (self.probe_card_cost_USD + self.test_station_cost_USD) / (5 * 365 * 24)  # 5-year amortization
        
        # Test cost
        test_cost = equipment_cost_per_hour * (test_time_minutes / 60)
        
        return test_cost
    
    def optimize_calibration_strategy(self, num_rings: int, target_test_time_min: float) -> Dict[str, Any]:
        """Optimize calibration strategy for target test time."""
        # Binary search for optimal calibration points
        min_points = 4
        max_points = 32
        
        best_points = min_points
        for points in range(min_points, max_points + 1):
            test_time = self.compute_test_time(num_rings, points)
            if test_time <= target_test_time_min:
                best_points = points
            else:
                break
        
        final_test_time = self.compute_test_time(num_rings, best_points)
        test_cost = self.compute_test_cost_per_die(final_test_time)
        
        return {
            "optimal_calibration_points": best_points,
            "test_time_minutes": final_test_time,
            "test_cost_per_die": test_cost,
            "throughput_dies_per_hour": 60 / final_test_time,
            "meets_target": final_test_time <= target_test_time_min
        }


def run_manufacturing_analysis(design_params: Dict[str, Any], 
                             array_size: Tuple[int, int] = (64, 64),
                             die_area_cm2: float = 0.25) -> Dict[str, Any]:
    """
    Run comprehensive manufacturing analysis for the accelerator design.
    
    Args:
        design_params: Design parameters from optimization
        array_size: Ring array dimensions
        die_area_cm2: Die area in cm²
        
    Returns:
        Comprehensive manufacturing analysis results
    """
    # Initialize models
    process_model = ProcessVariationModel()
    yield_model = YieldModel()
    corner_model = ProcessCornerModel()
    test_model = TestabilityModel()
    foundry_model = FoundryConstraints()
    reliability_model = ReliabilityModel()
    
    # Generate process variation maps
    spacing_um = design_params.get("ring_spacing_um", 10.0)
    cd_map = process_model.generate_cd_map(array_size, spacing_um)
    wavelength_shift_map = process_model.compute_wavelength_shift(cd_map)
    q_map = process_model.compute_q_factor_variation(cd_map)
    
    # Yield analysis
    yield_results = yield_model.compute_overall_yield(cd_map, q_map, wavelength_shift_map, 
                                                    die_area_cm2, array_size)
    
    # Process corner analysis
    corners_analysis = {}
    for corner in ["SS", "TT", "FF"]:
        corner_params = corner_model.apply_corner_effects(corner, design_params)
        corner_yield = corner_model.compute_corner_yield(yield_results["overall_yield"], corner)
        corners_analysis[corner] = {
            "yield": corner_yield,
            "parameters": corner_params
        }
    
    # Design rule compliance
    dr_score = foundry_model.compute_design_rule_score(design_params)
    dr_violations = foundry_model.validate_design_rules(design_params)
    
    # Test strategy optimization
    num_rings = array_size[0] * array_size[1]
    target_test_time = design_params.get("test_time_min", 2.0)
    test_strategy = test_model.optimize_calibration_strategy(num_rings, target_test_time)
    
    # Reliability assessment
    heater_current = design_params.get("heater_power_uW", 50) / 3.3  # Assume 3.3V
    max_temp = design_params.get("max_temp_C", 85)
    thermal_swing = design_params.get("thermal_swing_C", 20)
    reliability_results = reliability_model.assess_reliability(heater_current, max_temp, thermal_swing)
    
    # Comprehensive results
    manufacturing_analysis = {
        "process_variations": {
            "cd_map_statistics": {
                "mean_nm": float(np.mean(cd_map)),
                "std_nm": float(np.std(cd_map)),
                "min_nm": float(np.min(cd_map)),
                "max_nm": float(np.max(cd_map))
            },
            "wavelength_shift_statistics": {
                "mean_nm": float(np.mean(wavelength_shift_map)),
                "std_nm": float(np.std(wavelength_shift_map)),
                "max_abs_nm": float(np.max(np.abs(wavelength_shift_map)))
            },
            "q_factor_statistics": {
                "mean": float(np.mean(q_map)),
                "min": float(np.min(q_map)),
                "fraction_above_50k": float(np.sum(q_map >= 50000) / q_map.size)
            }
        },
        
        "yield_analysis": yield_results,
        "process_corners": corners_analysis,
        "design_rule_compliance": {
            "score": dr_score,
            "violations": dr_violations
        },
        "test_strategy": test_strategy,
        "reliability_assessment": reliability_results,
        
        "manufacturing_readiness": {
            "overall_score": (
                0.3 * yield_results["overall_yield"] * 100 +
                0.2 * dr_score +
                0.2 * (100 if test_strategy["meets_target"] else 0) +
                0.3 * (100 if reliability_results["meets_target"] else 0)
            ),
            "critical_issues": _identify_critical_issues(
                yield_results, dr_violations, test_strategy, reliability_results
            ),
            "recommendations": _generate_recommendations(
                yield_results, dr_violations, test_strategy, reliability_results
            )
        }
    }
    
    return manufacturing_analysis


def _identify_critical_issues(yield_results: Dict, dr_violations: Dict, 
                            test_strategy: Dict, reliability_results: Dict) -> List[str]:
    """Identify critical manufacturing issues."""
    issues = []
    
    if yield_results["overall_yield"] < 0.5:
        issues.append("Low overall yield (<50%)")
    
    if yield_results["functional_rings"] < 3200:
        issues.append("Insufficient functional rings for operation")
    
    if not all(dr_violations.values()):
        issues.append("Design rule violations detected")
    
    if not test_strategy["meets_target"]:
        issues.append("Test time exceeds target (>2 minutes)")
    
    if not reliability_results["meets_target"]:
        issues.append(f"Reliability limited by {reliability_results['limiting_mechanism']}")
    
    return issues


def _generate_recommendations(yield_results: Dict, dr_violations: Dict,
                            test_strategy: Dict, reliability_results: Dict) -> List[str]:
    """Generate manufacturing recommendations."""
    recommendations = []
    
    if yield_results["overall_yield"] < 0.7:
        recommendations.append("Consider relaxing Q-factor requirements or improving process control")
    
    if yield_results["yield_breakdown"]["wavelength_limited"] > 0.2:
        recommendations.append("Increase heater tuning range or improve CD control")
    
    if not all(dr_violations.values()):
        recommendations.append("Modify design to meet foundry design rules")
    
    if test_strategy["test_time_minutes"] > 2.0:
        recommendations.append("Reduce calibration points or parallelize testing")
    
    if reliability_results["limiting_mechanism"] == "electromigration":
        recommendations.append("Reduce heater current density or improve metallization")
    
    return recommendations


def export_manufacturing_specs(manufacturing_analysis: Dict[str, Any], 
                             output_dir: str = "manufacturing_specs") -> None:
    """Export manufacturing specifications for foundry."""
    Path(output_dir).mkdir(exist_ok=True)
    
    # Process specifications
    process_specs = {
        "wafer_specifications": {
            "substrate": "AlGaAs-on-Si",
            "wafer_size_mm": 150,
            "thickness_um": 200,
            "orientation": "(100)"
        },
        
        "lithography_specifications": {
            "critical_dimension_nm": manufacturing_analysis["process_variations"]["cd_map_statistics"]["mean_nm"],
            "cd_tolerance_nm": 3.0,  # ±3nm (3σ)
            "overlay_accuracy_nm": 20.0,
            "line_edge_roughness_nm": 2.0
        },
        
        "etch_specifications": {
            "etch_depth_nm": 220,
            "sidewall_angle_deg": 85,
            "selectivity_requirement": 50,
            "surface_roughness_nm": 1.0
        },
        
        "metallization_specifications": {
            "heater_metal": "TiN",
            "contact_metal": "Al",
            "sheet_resistance_ohm_sq": 50,
            "contact_resistance_ohm": 1e-6
        }
    }
    
    with open(f"{output_dir}/process_specifications.json", 'w') as f:
        json.dump(process_specs, f, indent=2)
    
    # Test specifications
    test_specs = {
        "wafer_level_test": {
            "probe_card_type": "optical",
            "test_temperature_range_C": [25, 85],
            "calibration_points": manufacturing_analysis["test_strategy"]["optimal_calibration_points"],
            "test_time_target_min": 2.0
        },
        
        "final_test": {
            "performance_benchmarks": [
                {"model": "ResNet-18", "min_tops": 2.0},
                {"model": "BERT-Base", "min_tops": 2.5},
                {"model": "Llama-7B", "min_tok_s": 40}
            ],
            "reliability_screening": {
                "burn_in_hours": 168,  # 1 week
                "temperature_C": 85,
                "voltage_stress": 1.1  # 10% overstress
            }
        }
    }
    
    with open(f"{output_dir}/test_specifications.json", 'w') as f:
        json.dump(test_specs, f, indent=2)
    
    print(f"🏭 Manufacturing specs exported to {output_dir}/")
