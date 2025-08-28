#!/usr/bin/env python
"""Debug the drift calculation issue."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def debug_drift():
    """Debug the drift calculation step by step."""
    
    # Values from the debug output
    P_high_W = 6e-05
    P_abs_W = 6.91e-09
    t_switch_ns = 1.4
    tau_thermal_ns = 60.0
    
    print("Debugging drift calculation:")
    print("=" * 60)
    
    # Convert to seconds
    t_s = t_switch_ns * 1e-9
    tau_th_s = tau_thermal_ns * 1e-9
    
    print(f"P_high_W: {P_high_W}")
    print(f"P_abs_W: {P_abs_W}")
    print(f"t_switch_ns: {t_switch_ns}")
    print(f"tau_thermal_ns: {tau_thermal_ns}")
    print()
    print(f"t_s (seconds): {t_s}")
    print(f"tau_th_s (seconds): {tau_th_s}")
    print()
    
    # Calculate components
    absorption_ratio = P_abs_W / P_high_W
    time_ratio = t_s / tau_th_s
    
    print(f"P_abs_W / P_high_W = {absorption_ratio:.6e}")
    print(f"t_s / tau_th_s = {time_ratio:.6e}")
    print()
    
    # Calculate drift
    drift = absorption_ratio * time_ratio
    print(f"drift = {drift:.6e}")
    print()
    
    # What if tau_thermal_ns was used directly as seconds?
    wrong_tau_th_s = tau_thermal_ns  # 60.0 seconds instead of 60e-9
    wrong_time_ratio = t_s / wrong_tau_th_s
    wrong_drift = absorption_ratio * wrong_time_ratio
    
    print("If tau_thermal_ns was wrongly used as seconds:")
    print(f"  wrong_tau_th_s: {wrong_tau_th_s}")
    print(f"  wrong_time_ratio: {wrong_time_ratio:.6e}")
    print(f"  wrong_drift: {wrong_drift:.6e}")
    print()
    
    # What if t_switch was not converted to seconds?
    wrong_t_s = t_switch_ns  # 1.4 instead of 1.4e-9
    wrong_time_ratio2 = wrong_t_s / tau_th_s
    wrong_drift2 = absorption_ratio * wrong_time_ratio2
    
    print("If t_switch_ns was not converted to seconds:")
    print(f"  wrong_t_s: {wrong_t_s}")
    print(f"  wrong_time_ratio: {wrong_time_ratio2:.6e}")
    print(f"  wrong_drift: {wrong_drift2:.6e}")
    print()
    
    # What if both were wrong?
    both_wrong_ratio = wrong_t_s / wrong_tau_th_s
    both_wrong_drift = absorption_ratio * both_wrong_ratio
    
    print("If both were wrong:")
    print(f"  time_ratio: {both_wrong_ratio:.6e}")
    print(f"  drift: {both_wrong_drift:.6e}")
    print()
    
    # The actual wrong value we're seeing
    actual_wrong = 5600
    print(f"Actual wrong drift from debug: {actual_wrong}")
    
    # Try to figure out what calculation gives 5600
    # 5600 = absorption_ratio * X
    # X = 5600 / absorption_ratio
    mystery_factor = actual_wrong / absorption_ratio
    print(f"Mystery factor X where drift = absorption_ratio * X:")
    print(f"  X = {mystery_factor:.2e}")
    print(f"  This is close to: {mystery_factor / 1e9:.2f} billion")
    
    # Check if this matches any combination
    print()
    print("Checking various combinations:")
    print(f"  t_switch_ns / tau_thermal_ns = {t_switch_ns / tau_thermal_ns:.6e}")
    print(f"  t_switch_ns / (tau_thermal_ns * 1e-9) = {t_switch_ns / (tau_thermal_ns * 1e-9):.2e}")
    print(f"  (t_switch_ns * 1e9) / tau_thermal_ns = {(t_switch_ns * 1e9) / tau_thermal_ns:.2e}")

if __name__ == "__main__":
    debug_drift()
