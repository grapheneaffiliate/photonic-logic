# photonic_logic_test_v11.py (minimal quick-check harness)
import numpy as np, pandas as pd
from dataclasses import dataclass
from typing import Dict

HBAR = 1.054571817e-34
C = 299792458.0
TWOPI = 2 * np.pi


@dataclass
class Params:
    kA: float
    kB: float
    J: float
    k_eA: float
    k_eB: float
    detA0: float
    detB0: float
    gXPM: float


def cm_transfer(params: Params, Pin_ctrl_W: float, dA_bias=0.0, dB_bias=0.0) -> Dict[str, float]:
    j = 1j
    dA = params.detA0 + dA_bias + params.gXPM * Pin_ctrl_W
    dB = params.detB0 + dB_bias
    M = np.array(
        [[j * dA - params.kA / 2, -1j * params.J], [-1j * params.J, j * dB - params.kB / 2]],
        dtype=complex,
    )
    bA = np.array([np.sqrt(params.k_eA), 0.0], dtype=complex)
    a = np.linalg.solve(M, bA)
    s_out = 1 - np.sqrt(params.k_eA) * a[0]
    T_thru = float(np.abs(s_out) ** 2)
    T_drop = float(np.abs(a[1]) ** 2)
    return {"T_through": T_thru, "T_drop": T_drop}


def truth_table(params: Params, ctrl_levels_W=(0.0, 1e-3)) -> pd.DataFrame:
    rows = []
    for P in ctrl_levels_W:
        T = cm_transfer(params, P)
        rows.append({"P_ctrl_W": P, **T})
    df = pd.DataFrame(rows)
    thr = 0.5 * (df.T_through.min() + df.T_through.max())
    df["logic_out"] = (df.T_through > thr).astype(int)
    df.attrs["threshold"] = thr
    return df


if __name__ == "__main__":
    GHz = TWOPI * 1e9
    params = Params(
        kA=0.39 * GHz,
        kB=0.39 * GHz,
        J=1.5 * GHz,
        k_eA=0.20 * GHz,
        k_eB=0.20 * GHz,
        detA0=0.0,
        detB0=0.0,
        gXPM=2.5 * GHz / 1e-3,
    )
    df = truth_table(params, (0.0, 0.5e-3, 1e-3, 2e-3))
    df.to_csv("truth_table.csv", index=False)
    print(df)
