#!/usr/bin/env python3
"""
Debug script to investigate why all Level 4 accelerator configurations are being rejected.
"""

import sys
import os
sys.path.append('DANTE')
sys.path.append('src')

from plogic.optimization.accelerator_system import PhotonicAcceleratorOptimizer
import numpy as np

def test_constraint_analysis():
    """Test individual constraints to find the issue."""
    
    # Create optimizer
    opt = PhotonicAcceleratorOptimizer()
    
    print("🔍 CONSTRAINT ANALYSIS DEBUG")
    print("=" * 50)
    
    # Test with a minimal valid configuration
    test_x = np.array([
        40, 40, 10.0,      # Ring geometry: 40x40 array, 10um spacing
        1550, 20, 15,      # Optical: 1550nm, 20mW/lane, 15dB budget  
        50, 6e-6, 10,      # Thermal: 50uW heater, 6us time constant, 10C gradient
        240, 0.8, 1,       # Manufacturing: 240nm CD, 80% yield, TT corner
        15, 1.0, 512,      # Architecture: 15 lanes, 1GHz clock, 512MB SRAM
        0.5, 0.2, 0.3,     # Power: 0.5W laser, 0.2W rings, 0.3W SRAM
        3.0, 50, 10,       # Performance: 3 TOPS, 50 tok/s, 10ms latency
        1, 2.0, 16, 1      # Integration: package type, 2min test, 16 cal points
    ])
    
    print("Test configuration:")
    print(f"  Input vector: {test_x}")
    
    # Extract parameters
    params = opt._extract_parameters(test_x)
    print(f"\nExtracted parameters:")
    for key, value in params.items():
        print(f"  {key}: {value}")
    
    # Test each constraint individually
    print(f"\n🔍 CONSTRAINT ANALYSIS:")
    
    # 1. Power constraint
    total_power = opt._compute_system_power(params)
    power_budget = opt.config.total_power_budget_W
    power_violation = total_power > power_budget
    print(f"1. Power: {total_power:.2f}W (limit: {power_budget}W) - {'❌ VIOLATION' if power_violation else '✅ OK'}")
    
    # 2. Thermal constraint
    peak_temp, thermal_feasible = opt._compute_thermal_performance(params)
    thermal_violation = peak_temp > 85.0
    print(f"2. Thermal: {peak_temp:.1f}°C (limit: 85°C) - {'❌ VIOLATION' if thermal_violation else '✅ OK'}")
    print(f"   Thermal feasible: {thermal_feasible}")
    
    # 3. Yield constraint
    yield_factor = opt._compute_manufacturing_yield(params)
    yield_violation = yield_factor < 0.5
    print(f"3. Yield: {yield_factor:.2f} (minimum: 0.5) - {'❌ VIOLATION' if yield_violation else '✅ OK'}")
    
    # 4. Performance constraint
    sustained_tops = opt._compute_sustained_performance(params)
    perf_violation = sustained_tops < 1.0
    print(f"4. Performance: {sustained_tops:.2f} TOPS (minimum: 1.0) - {'❌ VIOLATION' if perf_violation else '✅ OK'}")
    
    # Run full simulation
    result = opt._run_system_simulation(params)
    print(f"\n📊 FULL SIMULATION RESULTS:")
    print(f"  Valid config: {result['valid_config']}")
    print(f"  Total power: {result['total_power_W']:.2f}W")
    print(f"  Peak temp: {result['peak_temp_C']:.1f}°C")
    print(f"  Sustained TOPS: {result['sustained_tops']:.2f}")
    print(f"  Yield factor: {result['yield_factor']:.2f}")
    print(f"  Mobile score: {result['mobile_score']:.1f}")
    print(f"  Manufacturing score: {result['manufacturing_score']:.1f}")
    
    # Test final scoring
    score = opt(test_x, apply_scaling=False, track=False)
    print(f"\n🎯 FINAL SCORE: {score}")
    
    # Identify the root cause
    print(f"\n🔍 ROOT CAUSE ANALYSIS:")
    if power_violation:
        print(f"❌ POWER VIOLATION: {total_power:.2f}W > {power_budget}W")
        print(f"   Power breakdown:")
        print(f"     Laser: {params['laser_power_W']:.2f}W")
        print(f"     Rings: {params['ring_power_W']:.2f}W") 
        print(f"     SRAM: {params['sram_power_W']:.2f}W")
        print(f"     Fixed overhead: 0.8W (3R + ADC/DAC + Control)")
        print(f"     Total: {total_power:.2f}W")
    
    if thermal_violation:
        print(f"❌ THERMAL VIOLATION: {peak_temp:.1f}°C > 85°C")
    
    if yield_violation:
        print(f"❌ YIELD VIOLATION: {yield_factor:.2f} < 0.5")
    
    if perf_violation:
        print(f"❌ PERFORMANCE VIOLATION: {sustained_tops:.2f} < 1.0 TOPS")
    
    # Test with relaxed constraints
    print(f"\n🔧 TESTING WITH RELAXED CONSTRAINTS:")
    
    # Try with lower power configuration
    low_power_x = test_x.copy()
    low_power_x[15] = 0.4  # Reduce laser power
    low_power_x[16] = 0.1  # Reduce ring power
    low_power_x[17] = 0.2  # Reduce SRAM power
    
    params_relaxed = opt._extract_parameters(low_power_x)
    total_power_relaxed = opt._compute_system_power(params_relaxed)
    print(f"Relaxed power config: {total_power_relaxed:.2f}W")
    
    score_relaxed = opt(low_power_x, apply_scaling=False, track=False)
    print(f"Relaxed score: {score_relaxed}")
    
    return {
        "original_score": score,
        "relaxed_score": score_relaxed,
        "power_violation": power_violation,
        "thermal_violation": thermal_violation,
        "yield_violation": yield_violation,
        "performance_violation": perf_violation
    }

if __name__ == "__main__":
    results = test_constraint_analysis()
    print(f"\n✅ DEBUG COMPLETE")
    print(f"Results: {results}")
