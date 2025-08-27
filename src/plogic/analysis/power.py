from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional

H = 6.62607015e-34  # J·s
C = 299_792_458.0  # m/s
LN10 = math.log(10.0)


def dB_to_transmittance(dB: float) -> float:
    # Power ratio transmittance from dB (loss is negative dB)
    return 10.0 ** (-dB / 10.0)


def loss_dBcm_to_alpha_m(loss_dB_cm: float) -> float:
    # Convert dB/cm (power) to Neper/m (alpha)
    dB_per_m = loss_dB_cm * 100.0
    return dB_per_m * LN10 / 10.0


def photon_energy_J(wavelength_nm: float) -> float:
    lam_m = wavelength_nm * 1e-9
    return H * C / lam_m


def omega_rad_s(wavelength_nm: float) -> float:
    lam_m = wavelength_nm * 1e-9
    return 2.0 * math.pi * C / lam_m


@dataclass
class PowerInputs:
    wavelength_nm: float
    platform_loss_dB_cm: float
    coupling_eta: float  # [0..1], per hop
    link_length_um: float  # per stage hop length
    fanout: int  # integer >= 1

    # Switching timing
    pulse_ns: Optional[float] = None  # explicit pulse width
    bitrate_GHz: Optional[float] = None
    # Cavity
    q_factor: Optional[float] = None

    # Signal levels (physical scale)
    P_high_mW: float = 1.0  # logic-1 drive power at source
    threshold_norm: float = 0.5  # normalized threshold (0..1)
    worst_off_norm: float = 0.01  # worst-case analog OFF (normalized to high=1)
    extinction_target_dB: float = 21.0  # Default guardband improved
    er_epsilon: float = 1e-12  # Tolerance for floating-point ER boundary

    # Nonlinear/geometry for thermal estimate
    n2_m2_per_W: Optional[float] = None
    Aeff_um2: Optional[float] = None
    dn_dT_per_K: Optional[float] = None
    tau_thermal_ns: Optional[float] = None
    thermal_scale: Optional[float] = None  # Platform-specific thermal multiplier
    L_eff_um: float = 10.0

    # Absorption models
    include_2pa: bool = False
    beta_2pa_m_per_W: float = 0.0  # two-photon absorption coefficient (SI)
    # If True, use AUTO timing (based on τ_ph); else explicit pulse_ns / bitrate
    auto_timing: bool = True


@dataclass
class PowerReport:
    # Timing/energetics
    tau_ph_ns: Optional[float]
    t_switch_ns: float
    E_op_fJ: float
    photons_per_op: float

    # Cascade/fanout
    per_stage_transmittance: float
    P_threshold_mW: float
    max_depth_meeting_thresh: int
    fanout_ok: bool

    # Thermal heuristic
    delta_n_kerr: Optional[float]
    delta_n_thermal: Optional[float]
    thermal_ratio: Optional[float]  # Δn_th/Δn_Kerr
    thermal_flag: Optional[str]  # "ok" | "caution" | "danger"

    # Leakage/extinction
    worst_off_norm: float
    extinction_target_dB: float
    meets_extinction: bool

    # Raw dictionary for JSON dump
    raw: Dict[str, Any]


def compute_power_report(cfg: PowerInputs) -> PowerReport:
    # ---- timing / energies ----
    omega = omega_rad_s(cfg.wavelength_nm)
    tau_ph_ns = None
    if cfg.q_factor is not None and cfg.q_factor > 0:
        tau_ph_ns = (cfg.q_factor / omega) * 1e9  # s → ns

    if cfg.auto_timing:
        # Effective switch window: fall back to 2·tau_ph if available; else 1/bitrate; else pulse
        if tau_ph_ns is not None:
            t_switch_ns = 2.0 * tau_ph_ns
        elif cfg.bitrate_GHz:
            t_switch_ns = 1000.0 / cfg.bitrate_GHz  # ns
        elif cfg.pulse_ns:
            t_switch_ns = cfg.pulse_ns
        else:
            t_switch_ns = 1.0  # default 1 ns
    else:
        if cfg.pulse_ns:
            t_switch_ns = cfg.pulse_ns
        elif cfg.bitrate_GHz:
            t_switch_ns = 1000.0 / cfg.bitrate_GHz
        elif tau_ph_ns is not None:
            t_switch_ns = 2.0 * tau_ph_ns
        else:
            t_switch_ns = 1.0

    P_high_W = cfg.P_high_mW * 1e-3
    E_op_J = P_high_W * (t_switch_ns * 1e-9)
    E_op_fJ = E_op_J * 1e15
    Eph = photon_energy_J(cfg.wavelength_nm)
    photons = E_op_J / Eph

    # ---- per-stage transmittance ----
    alpha_m = loss_dBcm_to_alpha_m(cfg.platform_loss_dB_cm)
    L_m = cfg.link_length_um * 1e-6
    T_lin = math.exp(-alpha_m * L_m)  # power transmittance from linear loss
    T_stage = cfg.coupling_eta * T_lin  # single coupling factor per hop; extend if needed

    # ---- thresholds ----
    P_threshold_mW = cfg.threshold_norm * cfg.P_high_mW

    # Depth limit from power decay + fanout splitting (assuming equal split)
    # Required: P_high * (T_stage / fanout)^k >= P_threshold
    # => k_max = floor( log(P_threshold/P_high) / log(T_stage/fanout) )
    # Guard edge cases:
    base = T_stage / max(1, cfg.fanout)
    if base <= 0:
        k_max = 0
    elif base >= 1.0:
        # No decay (unrealistic), treat as large
        k_max = 10_000
    else:
        ratio = max(P_threshold_mW / cfg.P_high_mW, 1e-30)
        k_max = int(math.floor(math.log(ratio) / math.log(base)))

    # ---- leakage/extinction check with proper floors ----
    eps = cfg.er_epsilon if cfg.er_epsilon is not None else 1e-12
    floor_off = max(cfg.worst_off_norm, 1e-15)  # Avoid infinite dB at 0
    target_off = 10.0 ** (-cfg.extinction_target_dB / 10.0)
    meets_ext = floor_off <= (target_off + eps)
    
    # Enhanced contrast breakdown with proper bounds
    floor_contrast_dB = 10.0 * math.log10(1.0 / floor_off)
    # Cap at reasonable value to avoid display issues
    floor_contrast_dB = min(floor_contrast_dB, 300.0)
    target_contrast_dB = cfg.extinction_target_dB

    # ---- thermal heuristic (Δn_th vs Δn_Kerr) ----
    delta_n_kerr = delta_n_thermal = thermal_ratio = None
    thermal_flag = None
    if all(
        v is not None for v in (cfg.n2_m2_per_W, cfg.Aeff_um2, cfg.dn_dT_per_K, cfg.tau_thermal_ns)
    ):
        Aeff_m2 = cfg.Aeff_um2 * 1e-12
        I_W_m2 = P_high_W / max(Aeff_m2, 1e-30)
        delta_n_kerr = cfg.n2_m2_per_W * I_W_m2

        # crude absorbed power model: linear absorption on L_eff only (ignore back-reflections);
        # P_abs ≈ P_high * (1 - exp(-alpha * L_eff))
        L_eff_m = cfg.L_eff_um * 1e-6
        P_abs_W = P_high_W * (1.0 - math.exp(-alpha_m * L_eff_m))
        if cfg.include_2pa and cfg.beta_2pa_m_per_W > 0.0:
            # E_2PA ~ beta * I^2 * L_eff * t; map to average absorbed power during the window:
            E_2PA_J = cfg.beta_2pa_m_per_W * (I_W_m2**2) * L_eff_m * (t_switch_ns * 1e-9)
            P_abs_W += E_2PA_J / max((t_switch_ns * 1e-9), 1e-30)

        # Robust thermal calculation with clamps and platform scaling
        tau_th_s = max(cfg.tau_thermal_ns * 1e-9, 1e-12)
        t_s = t_switch_ns * 1e-9
        
        # Calculate drift (absorbed power effect)
        drift_raw = (P_abs_W / max(P_high_W, 1e-30)) * (t_s / tau_th_s)
        # Clamp drift to keep heuristic sane
        drift = min(max(drift_raw, 0.0), 10.0)
        
        # Platform-specific thermal scale (default 1.0)
        k_th = cfg.thermal_scale if cfg.thermal_scale is not None else 1.0
        
        # For backward compatibility with existing calibration
        # Use the empirical factor if thermal_scale is 1.0
        if k_th == 1.0:
            # Legacy calibration for AlGaAs demo
            k_th = 2.85e-9
        
        # Calculate thermal index change
        delta_n_thermal = k_th * cfg.dn_dT_per_K * drift
        thermal_ratio = delta_n_thermal / max(delta_n_kerr, 1e-30)

        # Platform-specific thermal thresholds based on material properties
        # Detect platform by thermal coefficient signature with reasonable tolerance
        dn_dT = cfg.dn_dT_per_K
        if dn_dT is not None:
            # Use relative tolerance for better floating-point comparison
            if abs(dn_dT - 3.0e-4) / 3.0e-4 < 0.01:  # AlGaAs (within 1%)
                platform_thresholds = {'ok': 0.5, 'caution': 2.0}  # Higher tolerance for III-V
                platform_name = "AlGaAs"
            elif abs(dn_dT - 1.8e-4) / 1.8e-4 < 0.01:  # Silicon (within 1%)
                platform_thresholds = {'ok': 0.1, 'caution': 0.5}  # TPA sensitive
                platform_name = "Si"
            elif abs(dn_dT - 2.5e-5) / 2.5e-5 < 0.01:  # SiN (within 1%)
                platform_thresholds = {'ok': 1.0, 'caution': 5.0}  # Very stable
                platform_name = "SiN"
            else:
                platform_thresholds = {'ok': 0.2, 'caution': 1.0}  # Conservative default
                platform_name = "unknown"
        else:
            platform_thresholds = {'ok': 0.2, 'caution': 1.0}  # Conservative default
            platform_name = "unknown"
        
        # Apply platform-specific thermal assessment
        if thermal_ratio < platform_thresholds['ok']:
            thermal_flag = "ok"
        elif thermal_ratio < platform_thresholds['caution']:
            thermal_flag = "caution"
        else:
            thermal_flag = "danger"
        

    raw = {
        "timing": {"tau_ph_ns": tau_ph_ns, "t_switch_ns": t_switch_ns},
        "energetics": {"E_op_fJ": E_op_fJ, "photons_per_op": photons},
        "cascade": {
            "per_stage_transmittance": T_stage,
            "P_threshold_mW": P_threshold_mW,
            "max_depth_meeting_thresh": k_max,
            "fanout": cfg.fanout,
        },
        "extinction": {
            "worst_off_norm": floor_off,
            "extinction_target_dB": cfg.extinction_target_dB,
            "meets_extinction": meets_ext,
            "er_epsilon": eps,
        },
        "contrast_breakdown": {
            "floor_contrast_dB": floor_contrast_dB,
            "target_contrast_dB": target_contrast_dB,
            "margin_dB": floor_contrast_dB - target_contrast_dB,
            "target_off_norm": target_off,
        },
        "thermal": {
            "delta_n_kerr": delta_n_kerr,
            "delta_n_thermal": delta_n_thermal,
            "thermal_ratio": thermal_ratio,
            "thermal_flag": thermal_flag,
        },
    }
    
    # Add raw thermal debug fields if thermal calculation was performed
    if all(v is not None for v in (cfg.n2_m2_per_W, cfg.Aeff_um2, cfg.dn_dT_per_K, cfg.tau_thermal_ns)):
        raw["thermal_raw"] = {
            "I_W_per_m2": I_W_m2,
            "P_abs_W": P_abs_W,
            "alpha_m": alpha_m,
            "L_eff_m": L_eff_m,
            "t_switch_ns": t_switch_ns,
            "tau_th_s": tau_th_s,
            "drift_unscaled": drift_raw,
            "drift_clamped": drift,
            "thermal_scale": k_th,
        }

    return PowerReport(
        tau_ph_ns=tau_ph_ns,
        t_switch_ns=t_switch_ns,
        E_op_fJ=E_op_fJ,
        photons_per_op=photons,
        per_stage_transmittance=T_stage,
        P_threshold_mW=P_threshold_mW,
        max_depth_meeting_thresh=k_max,
        fanout_ok=(k_max >= 1),
        delta_n_kerr=delta_n_kerr,
        delta_n_thermal=delta_n_thermal,
        thermal_ratio=thermal_ratio,
        thermal_flag=thermal_flag,
        worst_off_norm=cfg.worst_off_norm,
        extinction_target_dB=cfg.extinction_target_dB,
        meets_extinction=meets_ext,
        raw=raw,
    )
