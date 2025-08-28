"""
Hybrid material platform support for photonic logic.

This module provides support for hybrid Si-SiN platforms that combine:
- Silicon (Si) for high nonlinearity in logic gates
- Silicon Nitride (SiN) for low-loss routing
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np


@dataclass
class HybridMaterialConfig:
    """Configuration for hybrid material platforms."""
    
    # Primary materials
    logic_material: str = "Si"      # High χ³ for logic operations
    routing_material: str = "SiN"   # Low loss for interconnects
    
    # Coupling parameters
    coupling_efficiency: float = 0.95  # Si-SiN coupling efficiency
    transition_loss_dB: float = 0.1    # Loss per material transition
    
    # Material properties (from database.json equivalent)
    material_properties: Dict = None
    
    def __post_init__(self):
        """Initialize material properties."""
        if self.material_properties is None:
            self.material_properties = {
                "Si": {
                    "n": 3.48,
                    "n2": 4.5e-18,  # m²/W
                    "alpha_dB_per_cm": 2.0,
                    "beta_TPA": 5e-12,  # m/W
                    "thermal_coefficient": 1.86e-4,  # /K
                },
                "SiN": {
                    "n": 2.0,
                    "n2": 2.4e-19,  # m²/W - 20x lower than Si
                    "alpha_dB_per_cm": 0.1,  # 20x lower loss
                    "beta_TPA": 0,  # Negligible TPA
                    "thermal_coefficient": 2.45e-5,  # /K - more stable
                },
                "AlGaAs": {
                    "n": 3.3,
                    "n2": 2.6e-17,  # m²/W
                    "alpha_dB_per_cm": 1.0,
                    "beta_TPA": 2e-12,  # m/W
                    "thermal_coefficient": 4.5e-5,  # /K
                }
            }


class HybridPlatformOptimizer:
    """Optimizer for hybrid material platform designs."""
    
    def __init__(self, config: HybridMaterialConfig):
        self.config = config
        
    def calculate_segment_loss(self, 
                              length_mm: float, 
                              material: str,
                              power_mW: float = 10.0) -> float:
        """
        Calculate loss for a waveguide segment.
        
        Args:
            length_mm: Segment length in mm
            material: Material type ("Si", "SiN", etc.)
            power_mW: Optical power in mW
            
        Returns:
            Total loss in dB including linear and nonlinear contributions
        """
        props = self.config.material_properties.get(material, {})
        
        # Linear loss
        alpha_dB_per_mm = props.get("alpha_dB_per_cm", 1.0) / 10
        linear_loss = alpha_dB_per_mm * length_mm
        
        # Nonlinear loss (TPA) - only significant at high power
        beta_TPA = props.get("beta_TPA", 0)
        if beta_TPA > 0 and power_mW > 50:
            # Simplified TPA loss calculation
            P_W = power_mW / 1000
            A_eff = 0.1e-12  # Effective area in m²
            intensity = P_W / A_eff
            tpa_loss_per_mm = 4.343 * beta_TPA * intensity * 1e-3
            nonlinear_loss = tpa_loss_per_mm * length_mm
        else:
            nonlinear_loss = 0
            
        return linear_loss + nonlinear_loss
    
    def optimize_routing(self,
                        gate_positions: np.ndarray,
                        connections: list) -> Dict:
        """
        Optimize routing between gates using hybrid materials.
        
        Args:
            gate_positions: Nx2 array of gate (x,y) positions in mm
            connections: List of (source_idx, dest_idx) tuples
            
        Returns:
            Optimized routing plan with material assignments
        """
        routing_plan = {
            "segments": [],
            "total_loss_dB": 0,
            "material_usage": {"Si": 0, "SiN": 0},
            "transitions": 0
        }
        
        for src_idx, dst_idx in connections:
            src_pos = gate_positions[src_idx]
            dst_pos = gate_positions[dst_idx]
            distance = np.linalg.norm(dst_pos - src_pos)
            
            # Decision: Use Si for short connections, SiN for long ones
            if distance < 0.5:  # mm
                material = self.config.logic_material
                segment_type = "local"
            else:
                material = self.config.routing_material
                segment_type = "global"
                routing_plan["transitions"] += 2  # Entry and exit transitions
            
            # Calculate loss
            loss = self.calculate_segment_loss(distance, material)
            
            # Add transition losses if using SiN
            if material == self.config.routing_material:
                loss += 2 * self.config.transition_loss_dB
            
            segment = {
                "source": src_idx,
                "destination": dst_idx,
                "material": material,
                "type": segment_type,
                "distance_mm": distance,
                "loss_dB": loss
            }
            
            routing_plan["segments"].append(segment)
            routing_plan["total_loss_dB"] += loss
            routing_plan["material_usage"][material] += distance
            
        return routing_plan
    
    def calculate_thermal_stability(self, 
                                   material_distribution: Dict[str, float]) -> float:
        """
        Calculate thermal stability improvement from hybrid design.
        
        Args:
            material_distribution: Dict of material -> length_mm
            
        Returns:
            Thermal stability factor (higher is better)
        """
        total_length = sum(material_distribution.values())
        if total_length == 0:
            return 1.0
            
        weighted_coefficient = 0
        for material, length in material_distribution.items():
            props = self.config.material_properties.get(material, {})
            thermal_coeff = props.get("thermal_coefficient", 1e-4)
            weight = length / total_length
            weighted_coefficient += thermal_coeff * weight
            
        # Lower coefficient means better stability
        # Normalize to Si baseline
        si_coefficient = self.config.material_properties["Si"]["thermal_coefficient"]
        stability_factor = si_coefficient / weighted_coefficient
        
        return stability_factor


class HybridDesignRules:
    """Design rules for hybrid Si-SiN platforms."""
    
    @staticmethod
    def minimum_segment_length(material: str) -> float:
        """Minimum segment length to justify material transition."""
        rules = {
            "Si": 0.05,   # 50 μm - can be very short
            "SiN": 0.5,   # 500 μm - needs to be longer to justify transitions
            "AlGaAs": 0.1  # 100 μm
        }
        return rules.get(material, 0.1)
    
    @staticmethod
    def maximum_bend_radius(material: str) -> float:
        """Maximum bend radius in μm."""
        rules = {
            "Si": 5.0,     # Tight bends possible
            "SiN": 50.0,   # Larger bends needed
            "AlGaAs": 10.0
        }
        return rules.get(material, 10.0)
    
    @staticmethod
    def coupling_taper_length(from_material: str, to_material: str) -> float:
        """Required taper length for material transition in μm."""
        if from_material == to_material:
            return 0
        
        # Symmetric transitions
        transitions = {
            ("Si", "SiN"): 20.0,
            ("SiN", "Si"): 20.0,
            ("Si", "AlGaAs"): 15.0,
            ("AlGaAs", "Si"): 15.0,
            ("SiN", "AlGaAs"): 25.0,
            ("AlGaAs", "SiN"): 25.0,
        }
        return transitions.get((from_material, to_material), 20.0)


def demonstrate_hybrid_benefits():
    """Demonstrate the benefits of hybrid platform."""
    
    print("Hybrid Si-SiN Platform Analysis")
    print("=" * 50)
    
    # Create hybrid configuration
    config = HybridMaterialConfig()
    optimizer = HybridPlatformOptimizer(config)
    
    # Example: 4-gate circuit layout
    gate_positions = np.array([
        [0, 0],      # Gate 1
        [0.1, 0],    # Gate 2 (close - local routing)
        [2.0, 0],    # Gate 3 (far - global routing)
        [2.0, 1.5]   # Gate 4 (far - global routing)
    ])
    
    connections = [
        (0, 1),  # Local connection
        (1, 2),  # Global connection
        (2, 3),  # Global connection
    ]
    
    # Optimize routing
    routing = optimizer.optimize_routing(gate_positions, connections)
    
    print("\nRouting Optimization Results:")
    print("-" * 50)
    for segment in routing["segments"]:
        print(f"  {segment['source']} → {segment['destination']}: "
              f"{segment['material']} ({segment['type']}), "
              f"{segment['distance_mm']:.2f}mm, "
              f"Loss: {segment['loss_dB']:.2f}dB")
    
    print(f"\nTotal Routing Loss: {routing['total_loss_dB']:.2f} dB")
    print(f"Material Transitions: {routing['transitions']}")
    
    # Compare with Si-only
    si_only_loss = sum(
        optimizer.calculate_segment_loss(s['distance_mm'], 'Si') 
        for s in routing['segments']
    )
    
    print(f"\nComparison with Si-only:")
    print(f"  Si-only loss: {si_only_loss:.2f} dB")
    print(f"  Hybrid loss: {routing['total_loss_dB']:.2f} dB")
    print(f"  Improvement: {(1 - routing['total_loss_dB']/si_only_loss)*100:.1f}%")
    
    # Thermal stability
    stability = optimizer.calculate_thermal_stability(routing["material_usage"])
    print(f"\nThermal Stability Factor: {stability:.2f}x")
    
    # Design rules
    rules = HybridDesignRules()
    print("\nDesign Rules:")
    print(f"  Min SiN segment: {rules.minimum_segment_length('SiN')*1000:.0f} μm")
    print(f"  Si-SiN taper: {rules.coupling_taper_length('Si', 'SiN'):.0f} μm")
    print(f"  Max SiN bend radius: {rules.maximum_bend_radius('SiN'):.0f} μm")


if __name__ == "__main__":
    demonstrate_hybrid_benefits()
