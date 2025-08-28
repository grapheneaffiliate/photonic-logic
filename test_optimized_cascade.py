#!/usr/bin/env python
"""Test script to verify optimized cascade configuration achieves depth >= 30."""

import subprocess
import json
import sys

def test_optimized_cascade():
    """Test that optimized parameters achieve cascade depth >= 30."""
    
    # Run the cascade command with optimized parameters
    cmd = [
        "python", "-m", "plogic", "cascade",
        "--platform", "AlGaAs",
        "--P-high-mW", "0.06",
        "--pulse-ns", "1.4",
        "--coupling-eta", "0.98",
        "--link-length-um", "60",
        "--report", "power",
        "--beta", "80",
        "--threshold", "soft"
    ]
    
    print("Running command:", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Command failed with error:\n{result.stderr}")
        return False
    
    # Parse the output - it should be two JSON objects (cascade result and power report)
    output_lines = result.stdout.strip().split('\n')
    
    # Find the power report JSON (should be the second JSON object)
    power_report_json = None
    json_start = False
    json_buffer = []
    brace_count = 0
    
    for line in output_lines:
        if '{' in line:
            if not json_start:
                json_start = True
                json_buffer = []
            brace_count += line.count('{')
        
        if json_start:
            json_buffer.append(line)
            brace_count -= line.count('}')
            
            if brace_count == 0 and json_buffer:
                # Complete JSON object found
                try:
                    json_str = '\n'.join(json_buffer)
                    parsed = json.loads(json_str)
                    # Check if this is the power report (has 'cascade' key)
                    if 'cascade' in parsed:
                        power_report_json = parsed
                        break
                except json.JSONDecodeError:
                    pass
                json_start = False
                json_buffer = []
    
    if not power_report_json:
        print("Could not find power report in output")
        print("Raw output:", result.stdout)
        return False
    
    # Extract cascade depth
    cascade_info = power_report_json.get("cascade", {})
    max_depth = cascade_info.get("max_depth_meeting_thresh", 0)
    
    print(f"\n✓ Cascade depth achieved: {max_depth} stages")
    print(f"  Per-stage transmittance: {cascade_info.get('per_stage_transmittance', 0):.4f}")
    print(f"  P_threshold_mW: {cascade_info.get('P_threshold_mW', 0):.4f}")
    
    # Check thermal status
    thermal_info = power_report_json.get("thermal", {})
    thermal_flag = thermal_info.get("thermal_flag", "unknown")
    thermal_ratio = thermal_info.get("thermal_ratio", 0)
    print(f"\n✓ Thermal status: {thermal_flag}")
    print(f"  Thermal ratio: {thermal_ratio:.4f}")
    
    # Check energy efficiency
    energetics = power_report_json.get("energetics", {})
    E_op_fJ = energetics.get("E_op_fJ", 0)
    print(f"\n✓ Energy per operation: {E_op_fJ:.1f} fJ")
    
    # Verify cascade depth meets target
    if max_depth >= 30:
        print(f"\n🎉 SUCCESS: Achieved {max_depth} stages (target: ≥30)")
        return True
    else:
        print(f"\n❌ FAILED: Only achieved {max_depth} stages (target: ≥30)")
        return False

if __name__ == "__main__":
    success = test_optimized_cascade()
    sys.exit(0 if success else 1)
