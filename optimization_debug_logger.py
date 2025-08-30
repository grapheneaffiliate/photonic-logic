"""
Debug logging utility for photonic accelerator optimization.
Creates detailed CSV logs of all penalty components for debugging fake convergence issues.
"""

import csv
import os
from typing import Dict, Any
import numpy as np

class OptimizationDebugLogger:
    """Logger for detailed optimization debugging."""
    
    def __init__(self, log_file: str = "optimization_debug.csv"):
        self.log_file = log_file
        self.evaluation_count = 0
        
        # Create CSV with headers
        with open(self.log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "evaluation", "iteration", "total_power_W", "peak_temp_C", 
                "sustained_tops", "yield_factor", "primary_term",
                "p_power", "p_temp", "p_tops", "p_yield", 
                "total_penalty", "composite_score", "constraint_violations"
            ])
    
    def log_evaluation(self, iteration: int, params: Dict[str, Any], 
                      result: Dict[str, Any], penalties: Dict[str, float], 
                      score: float):
        """Log a single evaluation with all penalty components."""
        
        # Count constraint violations
        violations = 0
        if result["total_power_W"] > 2.0:
            violations += 1
        if result["peak_temp_C"] > 85.0:
            violations += 1
        if result["sustained_tops"] < 3.11:
            violations += 1
        if result["yield_factor"] < 0.5:
            violations += 1
        
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                self.evaluation_count,
                iteration,
                f"{result['total_power_W']:.4f}",
                f"{result['peak_temp_C']:.2f}",
                f"{result['sustained_tops']:.4f}",
                f"{result['yield_factor']:.4f}",
                f"{penalties['primary_term']:.4f}",
                f"{penalties['p_power']:.4f}",
                f"{penalties['p_temp']:.4f}",
                f"{penalties['p_tops']:.4f}",
                f"{penalties['p_yield']:.4f}",
                f"{penalties['total_penalty']:.4f}",
                f"{score:.6f}",
                violations
            ])
        
        self.evaluation_count += 1
    
    def print_summary(self):
        """Print summary statistics from the log."""
        if not os.path.exists(self.log_file):
            print("No debug log found.")
            return
        
        # Read the CSV and compute statistics
        scores = []
        penalties = []
        violations = []
        
        with open(self.log_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                scores.append(float(row['composite_score']))
                penalties.append(float(row['total_penalty']))
                violations.append(int(row['constraint_violations']))
        
        if not scores:
            print("No evaluations logged.")
            return
        
        scores = np.array(scores)
        penalties = np.array(penalties)
        violations = np.array(violations)
        
        print(f"\n📊 Optimization Debug Summary ({len(scores)} evaluations):")
        print(f"  Scores:     min={scores.min():.6f}, max={scores.max():.6f}, std={scores.std():.6f}")
        print(f"  Penalties:  min={penalties.min():.2f}, max={penalties.max():.2f}, std={penalties.std():.2f}")
        print(f"  Violations: {np.sum(violations > 0)} designs violate constraints ({100*np.sum(violations > 0)/len(violations):.1f}%)")
        print(f"  Unique scores: {len(np.unique(np.round(scores, 6)))}")
        
        if len(np.unique(np.round(scores, 6))) < 3:
            print("  ⚠️  WARNING: Very few unique scores - check penalty formulation!")
        
        if np.std(scores) < 1e-6:
            print("  ⚠️  WARNING: Scores have very low variance - likely constant!")


def test_penalty_sanity():
    """Test the penalty formulation with known constraint violations."""
    print("🧪 Testing penalty formulation sanity...")
    
    def relu(x: float) -> float:
        return x if x > 0 else 0.0
    
    # Test case 1: Massive power violation (like the original 106.76W)
    cap_power, cap_temp, min_tops, min_yield = 2.0, 85.0, 3.11, 0.5
    
    test_cases = [
        {"name": "Massive violation", "power": 106.76, "temp": 90.0, "tops": 16941.3, "yield": 0.35},
        {"name": "Moderate violation", "power": 5.0, "temp": 95.0, "tops": 2.0, "yield": 0.4},
        {"name": "Just feasible", "power": 1.9, "temp": 80.0, "tops": 3.2, "yield": 0.6},
        {"name": "Optimal", "power": 1.5, "temp": 70.0, "tops": 4.0, "yield": 0.8}
    ]
    
    for case in test_cases:
        primary = case["power"]
        p_power = relu(case["power"] - cap_power)
        p_temp = relu(case["temp"] - cap_temp)
        p_tops = relu(min_tops - case["tops"])
        p_yield = relu(min_yield - case["yield"])
        
        W_POWER, W_TEMP, W_TOPS, W_YIELD = 1000.0, 500.0, 200.0, 100.0
        
        total_penalty = (primary + W_POWER * p_power + W_TEMP * p_temp + 
                        W_TOPS * p_tops + W_YIELD * p_yield)
        score = 10000.0 / (total_penalty + 1.0)
        
        print(f"  {case['name']:15s}: penalty={total_penalty:8.1f}, score={score:.6f}")
        
        # Check that violations produce non-zero penalties
        if case["power"] > cap_power and p_power == 0:
            print(f"    ❌ Power penalty should be > 0 for {case['power']}W > {cap_power}W")
        if case["temp"] > cap_temp and p_temp == 0:
            print(f"    ❌ Temp penalty should be > 0 for {case['temp']}°C > {cap_temp}°C")
        if case["tops"] < min_tops and p_tops == 0:
            print(f"    ❌ TOPS penalty should be > 0 for {case['tops']} < {min_tops}")
        if case["yield"] < min_yield and p_yield == 0:
            print(f"    ❌ Yield penalty should be > 0 for {case['yield']} < {min_yield}")
    
    print("✅ Penalty sanity check complete.")


if __name__ == "__main__":
    test_penalty_sanity()
