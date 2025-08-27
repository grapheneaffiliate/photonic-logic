from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

C = 299_792_458.0  # m/s
LN10 = math.log(10.0)


def _dB_per_cm_to_alpha_m(loss_dB_per_cm: float) -> float:
    # Convert waveguide loss in dB/cm → Neper/m (power attenuation coeff alpha)
    # dB = 10*log10(Pout/Pin); alpha(Np/m) = (loss_dB_per_m) * ln(10)/10
    dB_per_m = loss_dB_per_cm * 100.0
    return dB_per_m * LN10 / 10.0


@dataclass
class Nonlinear:
    n2_m2_per_W: float
    beta_2pa_m_per_W: float
    gamma_Winv_km: Optional[float]
    group_index: float
    Aeff_um2_default: float


@dataclass
class Thermal:
    dn_dT_per_K: float
    thermal_conductivity_W_mK: float
    specific_heat_J_kgK: float
    density_kg_m3: float
    tau_thermal_ns: float


@dataclass
class Fabrication:
    Q_max: float
    loss_dB_per_cm: float
    coupling_tolerance_nm: float
    resonance_tolerance_pm: float


@dataclass
class Flags:
    cmos_compatible: bool
    tpa_present_at_1550: bool


@dataclass
class Platform:
    key: str
    name: str
    default_wavelength_nm: float
    nonlinear: Nonlinear
    thermal: Thermal
    fabrication: Fabrication
    flags: Flags

    def loss_alpha_m(self) -> float:
        """Power attenuation coefficient α [1/m] from dB/cm."""
        return _dB_per_cm_to_alpha_m(self.fabrication.loss_dB_per_cm)

    def gamma_Winv_m(
        self, wavelength_nm: Optional[float] = None, Aeff_um2: Optional[float] = None
    ) -> float:
        """
        Nonlinear coefficient γ [1/(W·m)].
        If database provides γ in W^-1 km, convert; else compute via γ ≈ n2 * ω / (c * Aeff).
        """
        if self.nonlinear.gamma_Winv_km is not None:
            return self.nonlinear.gamma_Winv_km / 1_000.0

        lam_nm = wavelength_nm or self.default_wavelength_nm
        lam_m = lam_nm * 1e-9
        omega = 2.0 * math.pi * C / lam_m
        Aeff_m2 = (Aeff_um2 or self.nonlinear.Aeff_um2_default) * 1e-12
        return self.nonlinear.n2_m2_per_W * omega / (C * Aeff_m2)

    def validate_reasonable_Q(self, intrinsic_Q: Optional[float]) -> Optional[str]:
        if intrinsic_Q is None:
            return None
        if intrinsic_Q > self.fabrication.Q_max * 1.5:
            return (
                f"Requested Q={intrinsic_Q:.2e} far exceeds platform Q_max={self.fabrication.Q_max:.2e}. "
                f"Expect yield issues or unrealistic assumptions."
            )
        return None


class PlatformDB:
    def __init__(self, path: Optional[Path] = None) -> None:
        default = Path(__file__).parent / "database.json"
        self.path = path or default
        with self.path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        self._platforms: Dict[str, Platform] = {}
        for k, v in raw["platforms"].items():
            self._platforms[k] = Platform(
                key=k,
                name=v["name"],
                default_wavelength_nm=v.get("default_wavelength_nm", 1550),
                nonlinear=Nonlinear(**v["nonlinear"]),
                thermal=Thermal(**v["thermal"]),
                fabrication=Fabrication(**v["fabrication"]),
                flags=Flags(**v["flags"]),
            )

    def get(self, key: str) -> Platform:
        if key not in self._platforms:
            raise KeyError(
                f"Unknown platform '{key}'. Available: {', '.join(self._platforms.keys())}"
            )
        return self._platforms[key]

    def keys(self):
        return list(self._platforms.keys())


def compute_gamma_from_params(n2_m2_per_W: float, wavelength_nm: float, Aeff_um2: float) -> float:
    lam_m = wavelength_nm * 1e-9
    omega = 2.0 * math.pi * C / lam_m
    return n2_m2_per_W * omega / (C * (Aeff_um2 * 1e-12))


def dB_contrast(on: float, off: float) -> float:
    if off <= 0.0:
        return float("inf")
    return 10.0 * math.log10(max(on, 1e-30) / max(off, 1e-30))
