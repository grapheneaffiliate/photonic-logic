#!/usr/bin/env python
"""Debug the absorption calculation."""

import sys
import os
import math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def debug_absorption():
    """Debug the absorption calculation step by step."""
    
    # Values from AlGaAs
    P_high_mW = 0.06
    P_high_W = P_high_mW * 1e-3
    loss_dB_cm = 0.5
    L_eff_um = 10.0
    L_eff_m = L_eff_um * 1e-6
    
    # For 2PA
    beta_2pa_m_per_W = 1.0e-10  # From AlGaAs
    Aeff_um2 = 0.5
    Aeff_m2 = Aeff_um2 * 1e-12
    t_switch_ns = 1.4
    
    print("Debugging absorption calculation:")
    print("=" * 60)
    
    # Linear absorption
    alpha_m = loss_dB_cm * 100.0 * math.log(10.0) / 10.0
    P_abs_linear = P_high_W * (1.0 - math.exp(-alpha_m * L_eff_m))
    
    print(f"P_high: {P_high_W} W")
    print(f"alpha: {alpha_m:.4f} /m")
    print(f"L_eff: {L_eff_m} m")
    print(f"Linear absorption: P_abs = {P_abs_linear:.2e} W")
    print()
    
    # 2PA calculation
    I_W_m2 = P_high_W / Aeff_m2
    print(f"Intensity: {I_W_m2:.2e} W/m²")
    print(f"beta_2pa: {beta_2pa_m_per_W} m/W")
    print()
    
    # Energy absorbed by 2PA
    E_2PA_J = beta_2pa_m_per_W * (I_W_m2**2) * L_eff_m * (t_switch_ns * 1e-9)
    print(f"E_2PA calculation:")
    print(f"  beta * I² = {beta_2pa_m_per_W * I_W_m2**2:.2e}")
    print(f"  * L_eff = {beta_2pa_m_per_W * I_W_m2**2 * L_eff_m:.2e}")
    print(f"  * t_switch = {E_2PA_J:.2e} J")
    
    # Average power during window
    P_2PA_W = E_2PA_J / (t_switch_ns * 1e-9)
    print(f"P_2PA = E_2PA / t_switch = {P_2PA_W:.2e} W")
    print()
    
    # Total absorbed power
    P_abs_total = P_abs_linear + P_2PA_W
    print(f"Total P_abs = {P_abs_total:.2e} W")
    print()
    
    # Check if there's a unit error
    print("Checking for unit errors:")
    print(f"  If t_switch_ns was not converted in denominator:")
    wrong_P_2PA = E_2PA_J / t_switch_ns  # Missing 1e-9
    print(f"    P_2PA = {wrong_P_2PA:.2e} W")
    print(f"    Total = {P_abs_linear + wrong_P_2PA:.2e} W")
    print()
    
    # Check if there's a double conversion
    print(f"  If t_switch was double-converted in numerator:")
    wrong_E_2PA = beta_2pa_m_per_W * (I_W_m2**2) * L_eff_m * (t_switch_ns * 1e-9 * 1e-9)
    wrong_P_2PA2 = wrong_E_2PA / (t_switch_ns * 1e-9)
    print(f"    E_2PA = {wrong_E_2PA:.2e} J")
    print(f"    P_2PA = {wrong_P_2PA2:.2e} W")
    print()
    
    # The actual wrong value we're seeing
    actual_wrong = 14.4
    print(f"Actual wrong P_abs from debug: {actual_wrong} W")
    print(f"This is {actual_wrong / P_abs_linear:.2e} times the linear absorption")
    print(f"This is {actual_wrong / P_high_W:.2e} times the input power")

if __name__ == "__main__":
    debug_absorption()
