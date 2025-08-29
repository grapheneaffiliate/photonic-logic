"""
Hybrid material platform support for photonic logic systems.

This module provides the HybridPlatform class for combining different materials
to leverage their specific advantages (e.g., AlGaAs for XPM logic, SiN for low-loss routing).
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from .platforms import PlatformDB


@dataclass
class HybridPlatform:
    """
    Hybrid material platform combining logic and routing materials.
    
    Attributes:
        logic_material: Material for logic operations (default: AlGaAs)
        routing_material: Material for low-loss routing (default: SiN)
        routing_fraction: Fraction of path length in routing material (0-1)
        prop_loss_logic_db_cm: Propagation loss for logic material (dB/cm)
        prop_loss_routing_db_cm: Propagation loss for routing material (dB/cm)
        mode_converter_loss_db: Loss per mode converter between materials (dB)
        coupling_efficiency: Efficiency of coupling between materials (0-1)
    """
    
    logic_material: str = 'AlGaAs'
    routing_material: str = 'SiN'
    routing_fraction: float = 0.5  # 50% routing by default
    prop_loss_logic_db_cm: float = 1.0  # AlGaAs typical: 1 dB/cm
    prop_loss_routing_db_cm: float = 0.1  # SiN typical: 0.1 dB/cm
    mode_converter_loss_db: float = 0.2  # Per transition
    coupling_efficiency: float = 0.95  # 95% coupling efficiency
    
    def __post_init__(self):
        """Initialize material platforms after dataclass initialization."""
        self.logic_platform = self._get_or_create_platform(self.logic_material)
        self.routing_platform = self._get_or_create_platform(self.routing_material)
        
        # Update propagation losses based on material properties if available
        if hasattr(self.logic_platform, 'prop_loss_db_cm'):
            self.prop_loss_logic_db_cm = self.logic_platform.prop_loss_db_cm
        if hasattr(self.routing_platform, 'prop_loss_db_cm'):
            self.prop_loss_routing_db_cm = self.routing_platform.prop_loss_db_cm
    
    def _get_or_create_platform(self, material_name: str):
        """Get platform from database or create a default one."""
        try:
            db = PlatformDB()
            return db.get(material_name)
        except Exception:
            # Create a simple object with necessary attributes if not in database
            class SimplePlatform:
                def __init__(self, name, prop_loss_db_cm, n_eff, n2):
                    self.name = name
                    self.prop_loss_db_cm = prop_loss_db_cm
                    self.n_eff = n_eff
                    self.n2 = n2
                    self.alpha_dB_cm = prop_loss_db_cm
            
            if material_name == 'SiN':
                return SimplePlatform(
                    name='SiN',
                    prop_loss_db_cm=0.1,  # Low propagation loss
                    n_eff=2.0,  # Silicon Nitride effective index
                    n2=2.4e-19  # Low Kerr coefficient
                )
            else:
                # Default to AlGaAs-like properties
                return SimplePlatform(
                    name=material_name,
                    prop_loss_db_cm=1.0,
                    n_eff=3.4,
                    n2=1e-17
                )
    
    def compute_transmittance(
        self, 
        link_length_um: float, 
        num_stages: int = 1,
        include_mode_converters: bool = True
    ) -> float:
        """
        Compute total transmittance through hybrid system.
        
        Args:
            link_length_um: Total link length in micrometers
            num_stages: Number of cascade stages
            include_mode_converters: Whether to include mode converter losses
            
        Returns:
            Total power transmittance (0-1)
        """
        # Convert to cm
        link_length_cm = link_length_um / 10000
        
        # Calculate losses in each section
        logic_length_cm = link_length_cm * (1 - self.routing_fraction)
        routing_length_cm = link_length_cm * self.routing_fraction
        
        # Propagation losses
        loss_logic_db = self.prop_loss_logic_db_cm * logic_length_cm * num_stages
        loss_routing_db = self.prop_loss_routing_db_cm * routing_length_cm * num_stages
        
        # Mode converter losses (2 per stage for transitions)
        converter_loss_db = 0
        if include_mode_converters and self.routing_fraction > 0 and self.routing_fraction < 1:
            converter_loss_db = 2 * self.mode_converter_loss_db * num_stages
        
        # Total loss
        total_loss_db = loss_logic_db + loss_routing_db + converter_loss_db
        
        # Convert to transmittance
        transmittance = 10 ** (-total_loss_db / 10)
        
        # Apply coupling efficiency
        if include_mode_converters:
            transmittance *= (self.coupling_efficiency ** (2 * num_stages))
        
        return transmittance
    
    def optimize_routing_fraction(
        self, 
        link_length_um: float, 
        num_stages: int = 1
    ) -> Tuple[float, float]:
        """
        Optimize routing fraction for minimum loss.
        
        Args:
            link_length_um: Total link length in micrometers
            num_stages: Number of cascade stages
            
        Returns:
            Tuple of (optimal_fraction, minimum_loss_db)
        """
        # Test different routing fractions
        fractions = np.linspace(0, 1, 101)
        losses = []
        
        for frac in fractions:
            self.routing_fraction = frac
            trans = self.compute_transmittance(link_length_um, num_stages)
            loss_db = -10 * np.log10(max(trans, 1e-30))
            losses.append(loss_db)
        
        # Find minimum
        min_idx = np.argmin(losses)
        optimal_fraction = fractions[min_idx]
        min_loss = losses[min_idx]
        
        # Restore original fraction
        self.routing_fraction = 0.5
        
        return optimal_fraction, min_loss
    
    def get_effective_parameters(self) -> Dict[str, float]:
        """
        Get effective parameters for the hybrid system.
        
        Returns:
            Dictionary of effective parameters
        """
        # Get n_eff from platforms (either from nonlinear.group_index or n_eff attribute)
        logic_n_eff = getattr(self.logic_platform, 'n_eff', 
                             getattr(self.logic_platform.nonlinear, 'group_index', 3.4) if hasattr(self.logic_platform, 'nonlinear') else 3.4)
        routing_n_eff = getattr(self.routing_platform, 'n_eff',
                               getattr(self.routing_platform.nonlinear, 'group_index', 2.0) if hasattr(self.routing_platform, 'nonlinear') else 2.0)
        
        # Get n2 from platforms
        logic_n2 = getattr(self.logic_platform, 'n2',
                          getattr(self.logic_platform.nonlinear, 'n2_m2_per_W', 1e-17) if hasattr(self.logic_platform, 'nonlinear') else 1e-17)
        routing_n2 = getattr(self.routing_platform, 'n2',
                            getattr(self.routing_platform.nonlinear, 'n2_m2_per_W', 2.4e-19) if hasattr(self.routing_platform, 'nonlinear') else 2.4e-19)
        
        # Weighted average of key parameters
        eff_n = logic_n_eff * (1 - self.routing_fraction) + routing_n_eff * self.routing_fraction
        eff_n2 = logic_n2 * (1 - self.routing_fraction) + routing_n2 * self.routing_fraction
        
        eff_loss = (self.prop_loss_logic_db_cm * (1 - self.routing_fraction) + 
                    self.prop_loss_routing_db_cm * self.routing_fraction)
        
        return {
            'effective_index': eff_n,
            'effective_n2': eff_n2,
            'effective_loss_db_cm': eff_loss,
            'logic_fraction': 1 - self.routing_fraction,
            'routing_fraction': self.routing_fraction,
            'logic_material': self.logic_material,
            'routing_material': self.routing_material,
            'mode_converter_loss_db': self.mode_converter_loss_db,
            'coupling_efficiency': self.coupling_efficiency
        }
    
    def design_cascade(
        self,
        target_depth: int = 33,
        gate_length_um: float = 100,
        routing_length_um: float = 500
    ) -> Dict[str, float]:
        """
        Design a cascade with hybrid routing.
        
        Args:
            target_depth: Target cascade depth
            gate_length_um: Length of each logic gate
            routing_length_um: Length of routing between gates
            
        Returns:
            Dictionary with cascade design parameters
        """
        total_link_um = gate_length_um + routing_length_um
        
        # Calculate routing fraction based on lengths
        actual_routing_fraction = routing_length_um / total_link_um
        
        # Calculate losses
        gate_loss_db = self.prop_loss_logic_db_cm * (gate_length_um / 10000)
        routing_loss_db = self.prop_loss_routing_db_cm * (routing_length_um / 10000)
        
        # Total loss per stage
        stage_loss_db = gate_loss_db + routing_loss_db + 2 * self.mode_converter_loss_db
        stage_transmittance = 10 ** (-stage_loss_db / 10) * (self.coupling_efficiency ** 2)
        
        # Maximum depth for 3dB total loss
        max_depth_3db = int(3.0 / stage_loss_db)
        
        # Total loss at target depth
        total_loss_db = stage_loss_db * target_depth
        total_transmittance = stage_transmittance ** target_depth
        
        return {
            'gate_length_um': gate_length_um,
            'routing_length_um': routing_length_um,
            'total_link_um': total_link_um,
            'routing_fraction': actual_routing_fraction,
            'stage_loss_db': stage_loss_db,
            'stage_transmittance': stage_transmittance,
            'max_depth_3db': max_depth_3db,
            'target_depth': target_depth,
            'total_loss_db': total_loss_db,
            'total_transmittance': total_transmittance,
            'improvement_factor': self.prop_loss_logic_db_cm / self.prop_loss_routing_db_cm
        }
    
    def __str__(self) -> str:
        """String representation of hybrid platform."""
        params = self.get_effective_parameters()
        return (
            f"HybridPlatform(\n"
            f"  Logic: {params['logic_material']} ({params['logic_fraction']:.0%})\n"
            f"  Routing: {params['routing_material']} ({params['routing_fraction']:.0%})\n"
            f"  Effective loss: {params['effective_loss_db_cm']:.2f} dB/cm\n"
            f"  Effective index: {params['effective_index']:.2f}\n"
            f"  Effective n2: {params['effective_n2']:.2e} m²/W\n"
            f")"
        )


def compare_platforms(
    single_material: str = 'AlGaAs',
    hybrid_config: Optional[HybridPlatform] = None,
    link_length_um: float = 600,
    num_stages: int = 33
) -> Dict[str, Dict[str, float]]:
    """
    Compare single-material vs hybrid platform performance.
    
    Args:
        single_material: Name of single material platform
        hybrid_config: HybridPlatform configuration (uses default if None)
        link_length_um: Total link length per stage
        num_stages: Number of cascade stages
        
    Returns:
        Dictionary comparing key metrics
    """
    if hybrid_config is None:
        hybrid_config = HybridPlatform()
    
    # Single material calculation
    single_platform = hybrid_config._get_or_create_platform(single_material)
    single_loss_db_cm = getattr(single_platform, 'prop_loss_db_cm', 
                                getattr(single_platform, 'alpha_dB_cm', 1.0))
    single_loss_total = single_loss_db_cm * (link_length_um / 10000) * num_stages
    single_trans = 10 ** (-single_loss_total / 10)
    
    # Hybrid calculation
    hybrid_trans = hybrid_config.compute_transmittance(link_length_um, num_stages)
    hybrid_loss_total = -10 * np.log10(max(hybrid_trans, 1e-30))
    
    # Optimized hybrid
    opt_fraction, opt_loss = hybrid_config.optimize_routing_fraction(link_length_um, num_stages)
    
    return {
        'single_material': {
            'material': single_material,
            'total_loss_db': single_loss_total,
            'transmittance': single_trans,
            'prop_loss_db_cm': single_loss_db_cm
        },
        'hybrid_default': {
            'config': f"{hybrid_config.logic_material}/{hybrid_config.routing_material}",
            'routing_fraction': hybrid_config.routing_fraction,
            'total_loss_db': hybrid_loss_total,
            'transmittance': hybrid_trans,
            'improvement_db': single_loss_total - hybrid_loss_total
        },
        'hybrid_optimized': {
            'optimal_routing_fraction': opt_fraction,
            'total_loss_db': opt_loss,
            'transmittance': 10 ** (-opt_loss / 10),
            'improvement_db': single_loss_total - opt_loss
        }
    }
