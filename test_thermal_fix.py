#!/usr/bin/env python3
"""
Test script to verify thermal flag consistency between cascade and demo commands.
"""

import json
import subprocess
import sys
from pathlib import Path

def run_command(cmd):
    """Run a command and return the output."""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=".")
        if result.returncode != 0:
            print(f"Error running command: {cmd}")
            print(f"stderr: {result.stderr}")
            return None
        return result.stdout
    except Exception as e:
        print(f"Exception running command: {cmd}")
        print(f"Error: {e}")
        return None

def extract_thermal_info(json_output):
    """Extract thermal information from JSON output."""
    try:
        data = json.loads(json_output)
        
        # For cascade command output
        if "thermal" in data:
            return data["thermal"]
        
        # For demo command output (nested structure)
        if "power_analysis" in data and "thermal" in data["power_analysis"]:
            return data["power_analysis"]["thermal"]
            
        return None
    except Exception as e:
        print(f"Error parsing JSON: {e}")
        return None

def main():
    print("🔍 Testing Thermal Flag Consistency")
    print("=" * 50)
    
    # Test parameters
    platform = "AlGaAs"
    power_mw = 0.1
    pulse_ns = 0.3
    
    print(f"Platform: {platform}")
    print(f"Power: {power_mw} mW")
    print(f"Pulse: {pulse_ns} ns")
    print()
    
    # Test cascade command
    print("📊 Testing CASCADE command...")
    cascade_cmd = f"plogic cascade --platform {platform} --P-high-mW {power_mw} --pulse-ns {pulse_ns} --report power"
    cascade_output = run_command(cascade_cmd)
    
    if cascade_output:
        # Parse the JSON output (cascade outputs two JSON objects)
        lines = cascade_output.strip().split('\n')
        json_lines = [line for line in lines if line.strip().startswith('{')]
        
        if len(json_lines) >= 2:
            # Second JSON object should be the power report
            cascade_thermal = extract_thermal_info(json_lines[1])
            if cascade_thermal:
                print(f"✅ CASCADE thermal_flag: {cascade_thermal.get('thermal_flag', 'N/A')}")
                print(f"   thermal_ratio: {cascade_thermal.get('thermal_ratio', 'N/A')}")
            else:
                print("❌ Could not extract thermal info from cascade")
        else:
            print("❌ Unexpected cascade output format")
    else:
        print("❌ CASCADE command failed")
    
    print()
    
    # Test demo command
    print("🚀 Testing DEMO command...")
    demo_cmd = f"plogic demo --platform {platform} --P-high-mW {power_mw} --pulse-ns {pulse_ns} --output json --report power"
    demo_output = run_command(demo_cmd)
    
    if demo_output:
        demo_thermal = extract_thermal_info(demo_output)
        if demo_thermal:
            print(f"✅ DEMO thermal_flag: {demo_thermal.get('thermal_flag', 'N/A')}")
            print(f"   thermal_ratio: {demo_thermal.get('thermal_ratio', 'N/A')}")
        else:
            print("❌ Could not extract thermal info from demo")
    else:
        print("❌ DEMO command failed")
    
    print()
    print("🎯 Expected Result for AlGaAs at 0.1 mW:")
    print("   thermal_flag: 'ok' (thermal_ratio should be ~5.8%, threshold is 50%)")

if __name__ == "__main__":
    main()
