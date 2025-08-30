"""
Photonic Logic Objective Functions for DANTE Optimization

This module implements DANTE-compatible objective functions for optimizing photonic logic circuits.
Supports single and multi-objective optimization across energy, cascade depth, thermal safety, and fabrication feasibility.
"""

import sys
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import numpy as np
import subprocess
import json

# Add DANTE to path for imports (with fallback for CI environments)
try:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'DANTE'))
    from dante.obj_functions import ObjectiveFunction
    from dante.utils import Tracker
    DANTE_AVAILABLE = True
except ImportError:
    # Fallback for environments without DANTE
    DANTE_AVAILABLE = False
    
    # Create minimal fallback classes
    class ObjectiveFunction(ABC):
        """Fallback ObjectiveFunction for when DANTE is not available."""
        def __init__(self):
            self.dims = 8
            self.turn = 0.01
            self.name = "fallback"
            self.lb = None
            self.ub = None
        
        def _preprocess(self, x):
            return np.array(x)
        
        @abstractmethod
        def __call__(self, x, apply_scaling=True, track=True):
            pass
    
    class Tracker:
        """Fallback Tracker for when DANTE is not available."""
        def __init__(self, name):
            self.name = name
            self.data = []
        
        def track(self, value, x):
            self.data.append({"value": value, "x": x})
        
        def track_metadata(self, metadata):
            pass

# Import photonic logic components
from ..controller import ExperimentController, PhotonicMolecule
from ..materials.platforms import PlatformDB


@dataclass
class PhotonicObjectiveBase(ObjectiveFunction):
    """Base class for photonic logic objective functions."""
    
    def _run_photonic_simulation(self, x: np.ndarray) -> Dict[str, Any]:
        """
        Run photonic logic simulation with parameters from DANTE.
        
        Args:
            x: Parameter vector from DANTE
            
        Returns:
            Simulation results dictionary
        """
        try:
            # Extract parameters with strict bounds enforcement
            platform_idx = int(np.clip(x[0], 0, 2))
            platforms = ["AlGaAs", "Si", "SiN"]
            platform = platforms[platform_idx]
            
            # Enforce minimum valid values to prevent invalid configurations
            p_high_mw = max(0.05, float(x[1]))  # Minimum 0.05 mW
            pulse_ns = max(0.05, float(x[2]))   # Minimum 0.05 ns
            coupling_eta = np.clip(float(x[3]), 0.7, 0.99)  # Realistic coupling range
            link_length_um = max(20.0, float(x[4]))  # Minimum 20 μm
            fanout = max(1, int(np.clip(x[5], 1, 8)))  # Force integer, minimum 1
            split_loss_db = max(0.2, float(x[6]))  # Minimum 0.2 dB
            stages = max(1, int(np.clip(x[7], 1, 50)))  # Force integer, minimum 1
            
            # Load platform configuration
            db = PlatformDB()
            platform_obj = db.get(platform)
            
            # Create device with platform-specific parameters
            dev = PhotonicMolecule(
                n2=platform_obj.nonlinear.n2_m2_per_W,
                xpm_mode="physics" if platform == "AlGaAs" else "linear"
            )
            
            # Run cascade simulation
            ctl = ExperimentController(dev)
            base_P_ctrl_W = p_high_mw * 1e-3
            pulse_duration_s = pulse_ns * 1e-9
            
            result = ctl.test_cascade(
                n_stages=stages,
                base_P_ctrl_W=base_P_ctrl_W,
                pulse_duration_s=pulse_duration_s
            )
            
            # Add configuration metadata
            for gate_type in result:
                result[gate_type]["platform"] = platform
                result[gate_type]["fanout"] = fanout
                result[gate_type]["split_loss_db"] = split_loss_db
                result[gate_type]["coupling_eta"] = coupling_eta
                result[gate_type]["link_length_um"] = link_length_um
                
                # Calculate fanout-adjusted metrics
                if fanout > 1:
                    effective_depth = max(1, int(stages / np.sqrt(fanout)))
                    split_efficiency = 10 ** (-split_loss_db / 10)
                    base_energy = result[gate_type].get("base_energy_fJ", base_P_ctrl_W * pulse_duration_s * 1e15)
                    result[gate_type]["effective_cascade_depth"] = effective_depth
                    result[gate_type]["split_efficiency"] = split_efficiency
                    result[gate_type]["fanout_adjusted_energy_fJ"] = base_energy * fanout
                else:
                    result[gate_type]["effective_cascade_depth"] = stages
                    result[gate_type]["fanout_adjusted_energy_fJ"] = result[gate_type].get("base_energy_fJ", base_P_ctrl_W * pulse_duration_s * 1e15)
            
            return result
            
        except Exception as e:
            # Return invalid result for failed simulations
            return {
                "XOR": {
                    "fanout_adjusted_energy_fJ": 1e6,  # Very high energy penalty
                    "effective_cascade_depth": 0,
                    "effective_P_ctrl_mW": 1e6,
                    "min_contrast_dB": 0,
                    "platform": "invalid"
                }
            }


@dataclass
class PhotonicEnergyOptimizer(PhotonicObjectiveBase):
    """Single-objective energy minimization for photonic logic circuits."""
    
    dims: int = 8
    turn: float = 0.01
    name: str = "photonic_energy"
    
    def __post_init__(self):
        # Parameter bounds: [platform_idx, P_high_mW, pulse_ns, coupling_eta, 
        #                   link_length_um, fanout, split_loss_db, stages]
        # Fixed bounds to prevent invalid configurations (no zeros)
        self.lb = np.array([0, 0.05, 0.05, 0.7, 20,  1, 0.2, 1])  # No zero pulse/stages
        self.ub = np.array([2, 5.0,  5.0,  0.99, 200, 8, 2.0, 20])
        self.tracker = Tracker(self.name + str(self.dims))
    
    def scaled(self, y: float) -> float:
        """Scale energy for DANTE (DANTE maximizes, so return 1/energy)."""
        return 10000 / (abs(y) + 1)
    
    def __call__(self, x: np.ndarray, apply_scaling: bool = True, track: bool = True) -> float:
        x = self._preprocess(x)
        result = self._run_photonic_simulation(x)
        
        # Extract energy from XOR gate (representative)
        energy_fJ = result["XOR"]["fanout_adjusted_energy_fJ"]
        
        # Apply penalties for impractical configurations
        power_mW = result["XOR"]["effective_P_ctrl_mW"]
        if power_mW > 100:  # >100mW impractical
            energy_fJ *= 100  # Heavy penalty
        
        if track:
            self.tracker.track(energy_fJ, x)
        
        # Return negative energy (DANTE maximizes, we want to minimize energy)
        return -energy_fJ if not apply_scaling else self.scaled(-energy_fJ)


@dataclass
class PhotonicCascadeOptimizer(PhotonicObjectiveBase):
    """Single-objective cascade depth maximization."""
    
    dims: int = 8
    turn: float = 0.01
    name: str = "photonic_cascade"
    
    def __post_init__(self):
        self.lb = np.array([0, 0.01, 0.05, 0.7, 5,   1, 0.1, 1])
        self.ub = np.array([2, 5.0,  2.0,  0.99, 200, 8, 2.0, 50])
        self.tracker = Tracker(self.name + str(self.dims))
    
    def scaled(self, y: float) -> float:
        """Scale cascade depth for DANTE."""
        return max(0, y) * 10  # Amplify for better optimization
    
    def __call__(self, x: np.ndarray, apply_scaling: bool = True, track: bool = True) -> float:
        x = self._preprocess(x)
        result = self._run_photonic_simulation(x)
        
        # Extract effective cascade depth
        cascade_depth = result["XOR"]["effective_cascade_depth"]
        
        # Apply penalties for poor performance
        contrast_dB = result["XOR"]["min_contrast_dB"]
        if contrast_dB < 10:  # Poor extinction ratio
            cascade_depth *= 0.1  # Heavy penalty
        
        if track:
            self.tracker.track(cascade_depth, x)
        
        return cascade_depth if not apply_scaling else self.scaled(cascade_depth)


@dataclass
class PhotonicThermalOptimizer(PhotonicObjectiveBase):
    """Single-objective thermal safety optimization."""
    
    dims: int = 8
    turn: float = 0.01
    name: str = "photonic_thermal"
    
    def __post_init__(self):
        self.lb = np.array([0, 0.01, 0.05, 0.7, 5,   1, 0.1, 1])
        self.ub = np.array([2, 5.0,  2.0,  0.99, 200, 8, 2.0, 20])
        self.tracker = Tracker(self.name + str(self.dims))
    
    def scaled(self, y: float) -> float:
        """Scale thermal safety score for DANTE."""
        return max(0, y)
    
    def __call__(self, x: np.ndarray, apply_scaling: bool = True, track: bool = True) -> float:
        x = self._preprocess(x)
        result = self._run_photonic_simulation(x)
        
        # Compute thermal safety score
        power_mW = result["XOR"]["effective_P_ctrl_mW"]
        platform = result["XOR"]["platform"]
        
        # Platform-specific thermal limits (mW)
        thermal_limits = {"AlGaAs": 1.0, "Si": 10.0, "SiN": 500.0}
        limit = thermal_limits.get(platform, 1.0)
        
        # Thermal safety score (0-100)
        if power_mW <= limit * 0.1:  # Well below limit
            thermal_score = 100
        elif power_mW >= limit:      # At or above limit
            thermal_score = 0
        else:
            thermal_score = 100 * (1 - (power_mW - limit*0.1) / (limit*0.9))
        
        if track:
            self.tracker.track(thermal_score, x)
        
        return thermal_score if not apply_scaling else self.scaled(thermal_score)


@dataclass
class PhotonicMultiObjective(PhotonicObjectiveBase):
    """Multi-objective optimization for photonic logic circuits."""
    
    dims: int = 12
    turn: float = 0.01
    name: str = "photonic_multi"
    
    # Objective weights
    energy_weight: float = 0.4
    cascade_weight: float = 0.3
    thermal_weight: float = 0.2
    fabrication_weight: float = 0.1
    
    def __post_init__(self):
        # Extended parameter space: [platform_idx, P_high_mW, pulse_ns, coupling_eta, 
        #                           link_length_um, fanout, split_loss_db, stages,
        #                           hybrid_flag, routing_fraction, include_2pa, auto_timing]
        self.lb = np.array([0, 0.01, 0.05, 0.7, 5,   1, 0.1, 1,  0, 0.1, 0, 0])
        self.ub = np.array([2, 5.0,  2.0,  0.99, 200, 8, 2.0, 50, 1, 0.9, 1, 1])
        self.tracker = Tracker(self.name + str(self.dims))
    
    def scaled(self, y: float) -> float:
        """Scale composite score for DANTE."""
        return max(0, y)
    
    def _compute_energy_score(self, result: Dict[str, Any]) -> float:
        """Compute energy efficiency score (0-100)."""
        energy_fJ = result["XOR"]["fanout_adjusted_energy_fJ"]
        
        # Logarithmic scoring: 10 fJ = 100 points, 1000 fJ = 0 points
        if energy_fJ <= 10:
            return 100
        elif energy_fJ >= 1000:
            return 0
        else:
            return 100 * (1 - np.log10(energy_fJ/10) / np.log10(100))
    
    def _compute_cascade_score(self, result: Dict[str, Any]) -> float:
        """Compute cascade performance score (0-100)."""
        effective_depth = result["XOR"]["effective_cascade_depth"]
        
        # Linear scoring: 1 stage = 0 points, 50 stages = 100 points
        return min(100, max(0, (effective_depth - 1) * 100 / 49))
    
    def _compute_thermal_score(self, result: Dict[str, Any]) -> float:
        """Compute thermal safety score (0-100)."""
        power_mW = result["XOR"]["effective_P_ctrl_mW"]
        platform = result["XOR"]["platform"]
        
        # Platform-specific thermal limits
        thermal_limits = {"AlGaAs": 1.0, "Si": 10.0, "SiN": 500.0}
        limit = thermal_limits.get(platform, 1.0)
        
        if power_mW <= limit * 0.1:  # Well below limit
            return 100
        elif power_mW >= limit:      # At or above limit
            return 0
        else:
            return 100 * (1 - (power_mW - limit*0.1) / (limit*0.9))
    
    def _compute_fabrication_score(self, result: Dict[str, Any]) -> float:
        """Compute fabrication feasibility score (0-100)."""
        platform = result["XOR"]["platform"]
        coupling_eta = result["XOR"].get("coupling_eta", 0.9)
        contrast_dB = result["XOR"]["min_contrast_dB"]
        
        # Platform maturity scores (CMOS compatibility)
        platform_scores = {"AlGaAs": 60, "Si": 100, "SiN": 90}
        base_score = platform_scores.get(platform, 50)
        
        # Coupling efficiency bonus (easier fabrication with looser tolerances)
        coupling_bonus = (1 - coupling_eta) * 20
        
        # Contrast penalty (need sufficient extinction ratio)
        contrast_bonus = min(20, contrast_dB - 10) if contrast_dB > 10 else -50
        
        return max(0, min(100, base_score + coupling_bonus + contrast_bonus))
    
    def _run_photonic_simulation(self, x: np.ndarray) -> Dict[str, Any]:
        """Extended simulation with hybrid platform support."""
        try:
            # Extract parameters
            platform_idx = int(np.clip(x[0], 0, 2))
            platforms = ["AlGaAs", "Si", "SiN"]
            platform = platforms[platform_idx]
            
            p_high_mw = float(x[1])
            pulse_ns = float(x[2])
            coupling_eta = float(x[3])
            link_length_um = float(x[4])
            fanout = int(np.clip(x[5], 1, 8))
            split_loss_db = float(x[6])
            stages = int(np.clip(x[7], 1, 50))
            
            # Extended parameters for multi-objective
            if len(x) >= 12:
                hybrid_flag = bool(x[8] > 0.5)
                routing_fraction = float(x[9])
                include_2pa = bool(x[10] > 0.5)
                auto_timing = bool(x[11] > 0.5)
            else:
                hybrid_flag = False
                routing_fraction = 0.5
                include_2pa = False
                auto_timing = False
            
            # Load platform configuration
            db = PlatformDB()
            platform_obj = db.get(platform)
            
            # Handle hybrid platform
            if hybrid_flag:
                from ..materials.hybrid import HybridPlatform
                hybrid_platform = HybridPlatform(
                    logic_material=platform,
                    routing_material="SiN",
                    routing_fraction=routing_fraction
                )
                eff_params = hybrid_platform.get_effective_parameters()
                effective_n2 = eff_params["effective_n2"]
                platform_name = f"Hybrid-{platform}/SiN"
            else:
                effective_n2 = platform_obj.nonlinear.n2_m2_per_W
                platform_name = platform
            
            # Create device
            dev = PhotonicMolecule(
                n2=effective_n2,
                xpm_mode="physics" if platform == "AlGaAs" else "linear"
            )
            
            # Run simulation
            ctl = ExperimentController(dev)
            base_P_ctrl_W = p_high_mw * 1e-3
            pulse_duration_s = pulse_ns * 1e-9
            
            result = ctl.test_cascade(
                n_stages=stages,
                base_P_ctrl_W=base_P_ctrl_W,
                pulse_duration_s=pulse_duration_s
            )
            
            # Add metadata
            for gate_type in result:
                result[gate_type]["platform"] = platform_name
                result[gate_type]["fanout"] = fanout
                result[gate_type]["split_loss_db"] = split_loss_db
                result[gate_type]["coupling_eta"] = coupling_eta
                result[gate_type]["link_length_um"] = link_length_um
                result[gate_type]["hybrid"] = hybrid_flag
                
                # Calculate fanout-adjusted metrics
                if fanout > 1:
                    effective_depth = max(1, int(stages / np.sqrt(fanout)))
                    split_efficiency = 10 ** (-split_loss_db / 10)
                    base_energy = result[gate_type].get("base_energy_fJ", base_P_ctrl_W * pulse_duration_s * 1e15)
                    result[gate_type]["effective_cascade_depth"] = effective_depth
                    result[gate_type]["split_efficiency"] = split_efficiency
                    result[gate_type]["fanout_adjusted_energy_fJ"] = base_energy * fanout
                else:
                    result[gate_type]["effective_cascade_depth"] = stages
                    result[gate_type]["fanout_adjusted_energy_fJ"] = result[gate_type].get("base_energy_fJ", base_P_ctrl_W * pulse_duration_s * 1e15)
            
            return result
            
        except Exception as e:
            # Return penalty result for failed simulations
            return {
                "XOR": {
                    "fanout_adjusted_energy_fJ": 1e6,
                    "effective_cascade_depth": 0,
                    "effective_P_ctrl_mW": 1e6,
                    "min_contrast_dB": 0,
                    "platform": "invalid",
                    "coupling_eta": 0.5,
                    "hybrid": False
                }
            }
    
    def __call__(self, x: np.ndarray, apply_scaling: bool = True, track: bool = True) -> float:
        x = self._preprocess(x)
        result = self._run_photonic_simulation(x)
        
        # Extract energy
        energy_fJ = result["XOR"]["fanout_adjusted_energy_fJ"]
        
        # Apply constraint penalties
        power_mW = result["XOR"]["effective_P_ctrl_mW"]
        if power_mW > 100:  # >100mW impractical
            energy_fJ *= 100
        
        if track:
            self.tracker.track(energy_fJ, x)
        
        # Return negative energy for minimization
        return -energy_fJ if not apply_scaling else self.scaled(-energy_fJ)


@dataclass
class PhotonicMultiObjective(PhotonicObjectiveBase):
    """Multi-objective optimization for photonic logic circuits."""
    
    dims: int = 12
    turn: float = 0.01
    name: str = "photonic_multi"
    
    # Objective weights (customizable)
    energy_weight: float = 0.4
    cascade_weight: float = 0.3
    thermal_weight: float = 0.2
    fabrication_weight: float = 0.1
    
    def __post_init__(self):
        # Extended parameter space for multi-objective
        self.lb = np.array([0, 0.01, 0.05, 0.7, 5,   1, 0.1, 1,  0, 0.1, 0, 0])
        self.ub = np.array([2, 5.0,  2.0,  0.99, 200, 8, 2.0, 50, 1, 0.9, 1, 1])
        self.tracker = Tracker(self.name + str(self.dims))
    
    def scaled(self, y: float) -> float:
        """Scale composite score for DANTE."""
        return max(0, y)
    
    def _compute_energy_score(self, result: Dict[str, Any]) -> float:
        """Compute energy efficiency score (0-100)."""
        energy_fJ = result["XOR"]["fanout_adjusted_energy_fJ"]
        
        # Logarithmic scoring: 10 fJ = 100 points, 1000 fJ = 0 points
        if energy_fJ <= 10:
            return 100
        elif energy_fJ >= 1000:
            return 0
        else:
            return 100 * (1 - np.log10(energy_fJ/10) / np.log10(100))
    
    def _compute_cascade_score(self, result: Dict[str, Any]) -> float:
        """Compute cascade performance score (0-100)."""
        effective_depth = result["XOR"]["effective_cascade_depth"]
        
        # Linear scoring: 1 stage = 0 points, 50 stages = 100 points
        return min(100, max(0, (effective_depth - 1) * 100 / 49))
    
    def _compute_thermal_score(self, result: Dict[str, Any]) -> float:
        """Compute thermal safety score (0-100)."""
        power_mW = result["XOR"]["effective_P_ctrl_mW"]
        platform = result["XOR"]["platform"]
        
        # Platform-specific thermal limits
        thermal_limits = {"AlGaAs": 1.0, "Si": 10.0, "SiN": 500.0}
        limit = thermal_limits.get(platform, 1.0)
        
        if power_mW <= limit * 0.1:  # Well below limit
            return 100
        elif power_mW >= limit:      # At or above limit
            return 0
        else:
            return 100 * (1 - (power_mW - limit*0.1) / (limit*0.9))
    
    def _compute_fabrication_score(self, result: Dict[str, Any]) -> float:
        """Compute fabrication feasibility score (0-100)."""
        platform = result["XOR"]["platform"]
        coupling_eta = result["XOR"].get("coupling_eta", 0.9)
        contrast_dB = result["XOR"]["min_contrast_dB"]
        
        # Platform maturity scores (CMOS compatibility)
        platform_scores = {"AlGaAs": 60, "Si": 100, "SiN": 90}
        base_score = platform_scores.get(platform, 50)
        
        # Coupling efficiency bonus (easier fabrication with looser tolerances)
        coupling_bonus = (1 - coupling_eta) * 20
        
        # Contrast penalty (need sufficient extinction ratio)
        contrast_bonus = min(20, contrast_dB - 10) if contrast_dB > 10 else -50
        
        return max(0, min(100, base_score + coupling_bonus + contrast_bonus))

    def __call__(self, x: np.ndarray, apply_scaling: bool = True, track: bool = True) -> float:
        x = self._preprocess(x)
        result = self._run_photonic_simulation(x)
        
        # Compute individual objective scores
        energy_score = self._compute_energy_score(result)
        cascade_score = self._compute_cascade_score(result)
        thermal_score = self._compute_thermal_score(result)
        fabrication_score = self._compute_fabrication_score(result)
        
        # Weighted composite score
        composite_score = (
            self.energy_weight * energy_score +
            self.cascade_weight * cascade_score +
            self.thermal_weight * thermal_score +
            self.fabrication_weight * fabrication_score
        )
        
        if track:
            self.tracker.track(composite_score, x)
            # Also track individual scores for analysis (if tracker supports metadata)
            try:
                self.tracker.track_metadata({
                    "energy_score": energy_score,
                    "cascade_score": cascade_score,
                    "thermal_score": thermal_score,
                    "fabrication_score": fabrication_score,
                    "energy_fJ": result["XOR"]["fanout_adjusted_energy_fJ"],
                    "cascade_depth": result["XOR"]["effective_cascade_depth"],
                    "power_mW": result["XOR"]["effective_P_ctrl_mW"],
                    "platform": result["XOR"]["platform"]
                })
            except AttributeError:
                # Tracker doesn't support metadata, skip
                pass
        
        return composite_score if not apply_scaling else self.scaled(composite_score)


def create_photonic_optimizer(
    objective_type: str = "multi",
    energy_weight: float = 0.4,
    cascade_weight: float = 0.3,
    thermal_weight: float = 0.2,
    fabrication_weight: float = 0.1,
    dims: int = 12
) -> PhotonicObjectiveBase:
    """
    Factory function to create photonic optimizers.
    
    Args:
        objective_type: Type of optimizer ("energy", "cascade", "thermal", "multi")
        energy_weight: Weight for energy objective (multi-objective only)
        cascade_weight: Weight for cascade objective (multi-objective only)
        thermal_weight: Weight for thermal objective (multi-objective only)
        fabrication_weight: Weight for fabrication objective (multi-objective only)
        dims: Number of optimization dimensions
        
    Returns:
        Configured photonic optimizer
    """
    if objective_type == "energy":
        return PhotonicEnergyOptimizer(dims=min(dims, 8))
    elif objective_type == "cascade":
        return PhotonicCascadeOptimizer(dims=min(dims, 8))
    elif objective_type == "thermal":
        return PhotonicThermalOptimizer(dims=min(dims, 8))
    elif objective_type == "multi":
        return PhotonicMultiObjective(
            dims=dims,
            energy_weight=energy_weight,
            cascade_weight=cascade_weight,
            thermal_weight=thermal_weight,
            fabrication_weight=fabrication_weight
        )
    else:
        raise ValueError(f"Unknown objective type: {objective_type}")


def run_photonic_optimization(
    objective_type: str = "multi",
    num_initial_samples: int = 50,
    num_acquisitions: int = 200,
    samples_per_acquisition: int = 10,
    **kwargs
) -> Dict[str, Any]:
    """
    Run DANTE optimization for photonic logic circuits.
    
    Args:
        objective_type: Type of optimization ("energy", "cascade", "thermal", "multi")
        num_initial_samples: Number of initial random samples
        num_acquisitions: Number of DANTE acquisition iterations
        samples_per_acquisition: Number of samples per acquisition
        **kwargs: Additional arguments for optimizer configuration
        
    Returns:
        Optimization results including best configurations and Pareto front
    """
    # Import DANTE components
    from dante.neural_surrogate import AckleySurrogateModel  # We'll adapt this
    from dante.tree_exploration import TreeExploration
    from dante.utils import generate_initial_samples
    
    # Create objective function
    obj_function = create_photonic_optimizer(objective_type, **kwargs)
    
    # Create surrogate model (adapt for photonic optimization)
    surrogate = AckleySurrogateModel(input_dims=obj_function.dims, epochs=100)
    
    # Generate initial samples
    input_x, input_y = generate_initial_samples(
        obj_function, num_init_samples=num_initial_samples, apply_scaling=True
    )
    
    # Track best solutions
    best_solutions = []
    
    # Main optimization loop
    for i in range(num_acquisitions):
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
        
        # Track progress
        best_idx = np.argmax(input_y)
        best_solutions.append({
            "iteration": i,
            "best_score": input_y[best_idx],
            "best_params": input_x[best_idx].tolist(),
            "num_evaluations": len(input_y)
        })
        
        print(f"Iteration {i+1}/{num_acquisitions}: Best score = {input_y[best_idx]:.4f}, Evaluations = {len(input_y)}")
        
        # Early stopping for convergence
        if i > 10 and len(set(input_y[-10:])) < 3:  # No improvement in last 10 iterations
            print(f"Converged after {i+1} iterations")
            break
    
    # Return optimization results
    best_idx = np.argmax(input_y)
    return {
        "best_score": input_y[best_idx],
        "best_parameters": input_x[best_idx],
        "optimization_history": best_solutions,
        "all_evaluations": {"x": input_x, "y": input_y},
        "objective_type": objective_type,
        "total_evaluations": len(input_y)
    }
