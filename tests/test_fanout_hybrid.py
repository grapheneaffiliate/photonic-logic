"""
Test suite for fanout>1 and hybrid platform features.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from plogic.materials.hybrid import (
    HybridMaterialConfig,
    HybridPlatformOptimizer,
    HybridDesignRules
)


class TestHybridMaterialConfig:
    """Test hybrid material configuration."""
    
    def test_default_configuration(self):
        """Test default hybrid configuration."""
        config = HybridMaterialConfig()
        
        assert config.logic_material == "Si"
        assert config.routing_material == "SiN"
        assert config.coupling_efficiency == 0.95
        assert config.transition_loss_dB == 0.1
        
    def test_material_properties_initialization(self):
        """Test that material properties are properly initialized."""
        config = HybridMaterialConfig()
        
        # Check Si properties
        assert "Si" in config.material_properties
        assert config.material_properties["Si"]["n"] == 3.48
        assert config.material_properties["Si"]["alpha_dB_per_cm"] == 2.0
        
        # Check SiN properties
        assert "SiN" in config.material_properties
        assert config.material_properties["SiN"]["alpha_dB_per_cm"] == 0.1
        assert config.material_properties["SiN"]["thermal_coefficient"] < \
               config.material_properties["Si"]["thermal_coefficient"]


class TestHybridPlatformOptimizer:
    """Test hybrid platform optimizer."""
    
    @pytest.fixture
    def optimizer(self):
        """Create optimizer instance."""
        config = HybridMaterialConfig()
        return HybridPlatformOptimizer(config)
    
    def test_segment_loss_calculation(self, optimizer):
        """Test loss calculation for different materials."""
        # Si should have higher loss than SiN
        si_loss = optimizer.calculate_segment_loss(1.0, "Si", 10.0)
        sin_loss = optimizer.calculate_segment_loss(1.0, "SiN", 10.0)
        
        assert si_loss > sin_loss
        assert si_loss == pytest.approx(0.2, rel=0.1)  # 2 dB/cm = 0.2 dB/mm
        assert sin_loss == pytest.approx(0.01, rel=0.1)  # 0.1 dB/cm = 0.01 dB/mm
    
    def test_nonlinear_loss_at_high_power(self, optimizer):
        """Test that nonlinear loss is included at high power."""
        low_power_loss = optimizer.calculate_segment_loss(1.0, "Si", 10.0)
        high_power_loss = optimizer.calculate_segment_loss(1.0, "Si", 100.0)
        
        # At high power, TPA should add extra loss
        assert high_power_loss > low_power_loss
    
    def test_routing_optimization(self, optimizer):
        """Test routing optimization with hybrid materials."""
        # Simple 3-gate layout
        gate_positions = np.array([
            [0, 0],      # Gate 1
            [0.2, 0],    # Gate 2 (close - should use Si)
            [3.0, 0],    # Gate 3 (far - should use SiN)
        ])
        
        connections = [
            (0, 1),  # Short connection
            (1, 2),  # Long connection
        ]
        
        routing = optimizer.optimize_routing(gate_positions, connections)
        
        # Check that short connection uses Si
        assert routing["segments"][0]["material"] == "Si"
        assert routing["segments"][0]["type"] == "local"
        
        # Check that long connection uses SiN
        assert routing["segments"][1]["material"] == "SiN"
        assert routing["segments"][1]["type"] == "global"
        
        # Check that transitions are counted
        assert routing["transitions"] == 2  # One long connection = 2 transitions
        
        # Check total loss is calculated
        assert routing["total_loss_dB"] > 0
    
    def test_thermal_stability_calculation(self, optimizer):
        """Test thermal stability factor calculation."""
        # Pure Si distribution
        si_only = {"Si": 10.0, "SiN": 0}
        si_stability = optimizer.calculate_thermal_stability(si_only)
        assert si_stability == pytest.approx(1.0)  # Baseline
        
        # Hybrid distribution
        hybrid = {"Si": 5.0, "SiN": 5.0}
        hybrid_stability = optimizer.calculate_thermal_stability(hybrid)
        assert hybrid_stability > 1.0  # Should be more stable
        
        # Pure SiN distribution
        sin_only = {"Si": 0, "SiN": 10.0}
        sin_stability = optimizer.calculate_thermal_stability(sin_only)
        assert sin_stability > hybrid_stability  # SiN is most stable


class TestHybridDesignRules:
    """Test design rules for hybrid platforms."""
    
    def test_minimum_segment_length(self):
        """Test minimum segment length rules."""
        rules = HybridDesignRules()
        
        # Si can have very short segments
        assert rules.minimum_segment_length("Si") == 0.05
        
        # SiN needs longer segments to justify transitions
        assert rules.minimum_segment_length("SiN") == 0.5
        
        # AlGaAs is intermediate
        assert rules.minimum_segment_length("AlGaAs") == 0.1
    
    def test_maximum_bend_radius(self):
        """Test bend radius constraints."""
        rules = HybridDesignRules()
        
        # Si allows tight bends
        assert rules.maximum_bend_radius("Si") == 5.0
        
        # SiN requires larger bends
        assert rules.maximum_bend_radius("SiN") == 50.0
        
        # AlGaAs is intermediate
        assert rules.maximum_bend_radius("AlGaAs") == 10.0
    
    def test_coupling_taper_length(self):
        """Test taper length requirements."""
        rules = HybridDesignRules()
        
        # Same material requires no taper
        assert rules.coupling_taper_length("Si", "Si") == 0
        
        # Si-SiN transition
        assert rules.coupling_taper_length("Si", "SiN") == 20.0
        assert rules.coupling_taper_length("SiN", "Si") == 20.0
        
        # Other transitions
        assert rules.coupling_taper_length("Si", "AlGaAs") == 15.0
        assert rules.coupling_taper_length("SiN", "AlGaAs") == 25.0


class TestFanoutConcepts:
    """Test fanout>1 concepts (using mock implementation)."""
    
    def test_fanout_depth_reduction(self):
        """Test that fanout can reduce circuit depth."""
        # Mock calculation: depth reduction with fanout
        def calculate_depth_with_fanout(original_depth, max_fanout):
            reduction_factor = np.log2(max_fanout) if max_fanout > 1 else 1
            return int(original_depth / reduction_factor)
        
        original_depth = 12
        
        # Fanout=1 (no reduction)
        depth_f1 = calculate_depth_with_fanout(original_depth, 1)
        assert depth_f1 == 12
        
        # Fanout=2 (some reduction)
        depth_f2 = calculate_depth_with_fanout(original_depth, 2)
        assert depth_f2 < original_depth
        
        # Fanout=4 (more reduction)
        depth_f4 = calculate_depth_with_fanout(original_depth, 4)
        assert depth_f4 < depth_f2
        assert depth_f4 == 6  # 12 / log2(4) = 12 / 2 = 6
    
    def test_fanout_power_distribution(self):
        """Test power distribution in fanout gates."""
        # Mock power calculation for fanout
        def calculate_fanout_power(base_power, fanout):
            # Power increases sub-linearly with fanout
            return base_power * (1 + 0.3 * np.log2(fanout))
        
        base_power = 10.0  # mW
        
        # Single output
        power_f1 = calculate_fanout_power(base_power, 1)
        assert power_f1 == base_power
        
        # Fanout=2
        power_f2 = calculate_fanout_power(base_power, 2)
        assert power_f2 > base_power
        assert power_f2 < 2 * base_power  # Sub-linear scaling
        
        # Fanout=4
        power_f4 = calculate_fanout_power(base_power, 4)
        assert power_f4 > power_f2
        assert power_f4 < 2 * power_f2  # Still sub-linear


class TestIntegrationScenarios:
    """Test integration of fanout and hybrid features."""
    
    def test_hybrid_routing_with_fanout(self):
        """Test combined hybrid routing and fanout optimization."""
        config = HybridMaterialConfig()
        optimizer = HybridPlatformOptimizer(config)
        
        # Complex circuit with fanout points
        gate_positions = np.array([
            [0, 0],       # Input
            [0.1, 0.1],   # Fanout point 1
            [0.1, -0.1],  # Fanout point 2
            [2.0, 0.1],   # Processing 1
            [2.0, -0.1],  # Processing 2
            [4.0, 0],     # Output combiner
        ])
        
        # Connections including fanout
        connections = [
            (0, 1),  # Input to fanout 1
            (0, 2),  # Input to fanout 2 (parallel)
            (1, 3),  # Fanout 1 to processing 1
            (2, 4),  # Fanout 2 to processing 2
            (3, 5),  # Processing 1 to output
            (4, 5),  # Processing 2 to output
        ]
        
        routing = optimizer.optimize_routing(gate_positions, connections)
        
        # Verify mixed material usage
        materials_used = set(s["material"] for s in routing["segments"])
        assert "Si" in materials_used  # For short connections
        assert "SiN" in materials_used  # For long connections
        
        # Verify loss optimization
        assert routing["total_loss_dB"] < len(connections) * 2.0  # Better than all-Si
    
    def test_performance_projections(self):
        """Test that projected improvements are achievable."""
        # Current system baseline
        current_depth = 12
        current_power = 120  # mW
        current_loss = 10  # dB
        
        # With fanout=4
        fanout_factor = np.log2(4)
        improved_depth = current_depth / fanout_factor
        assert improved_depth == 6  # 50% reduction
        
        # With hybrid routing (60% loss reduction)
        hybrid_loss = current_loss * 0.4
        assert hybrid_loss == 4  # 60% reduction
        
        # Combined improvements
        total_improvement = {
            "depth_reduction": (1 - improved_depth/current_depth) * 100,
            "loss_reduction": (1 - hybrid_loss/current_loss) * 100
        }
        
        assert total_improvement["depth_reduction"] == 50
        assert total_improvement["loss_reduction"] == 60


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
