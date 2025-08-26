from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path
from typing import List, Optional

import pandas as pd
import typer

from .controller import (
    ExperimentController,
    PhotonicMolecule,
    generate_design_report,
)
from .utils import soft_logic

# Keep help string consistent with smoke test expectations
app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Programmable Photonic Logic CLI",
)


@app.callback(invoke_without_command=True)
def main_callback(
    ctx: typer.Context,
    version: bool = typer.Option(False, "--version", help="Show version and exit"),
) -> None:
    """Programmable Photonic Logic CLI."""
    if version:
        try:
            v = importlib.metadata.version("photonic-logic")
            typer.echo(v)
        except importlib.metadata.PackageNotFoundError:
            typer.echo("2.2.0")  # fallback for development
        raise typer.Exit()

    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())


@app.command("characterize")
def characterize(
    stages: int = typer.Option(2, "--stages", help="Cascade stages for the demo"),
    report: Path = typer.Option(
        Path("photonic_logic_report.json"),
        "--report",
        help="Output JSON report path",
    ),
    threshold: str = typer.Option(
        "hard", "--threshold", help="Thresholding mode: 'hard' or 'soft'"
    ),
    beta: float = typer.Option(25.0, "--beta", help="Sigmoid slope (soft mode)"),
    xpm_mode: str = typer.Option(
        "linear", "--xpm-mode", help="XPM model: 'linear' (default) or 'physics'"
    ),
    n2: Optional[float] = typer.Option(
        None, "--n2", help="Kerr coefficient n2 (m^2/W) for physics XPM mode"
    ),
    a_eff: float = typer.Option(
        0.6e-12, "--a-eff", help="Effective mode area A_eff (m^2) for physics mode"
    ),
    n_eff: float = typer.Option(3.4, "--n-eff", help="Effective index n_eff"),
    g_geom: float = typer.Option(1.0, "--g-geom", help="Geometry scaling g_geom"),
) -> None:
    """
    Run default characterization and save report JSON.
    """
    dev = PhotonicMolecule(xpm_mode=xpm_mode, n2=n2, A_eff=a_eff, n_eff=n_eff, g_geom=g_geom)
    ctl = ExperimentController(dev)
    ctl.run_full_characterization()
    ctl.results["cascade"] = ctl.test_cascade(n_stages=stages, threshold_mode=threshold, beta=beta)
    rep = generate_design_report(dev, ctl.results, filename=str(report))
    typer.echo(json.dumps(rep, indent=2))
    typer.echo(f"Saved report to {report}")


@app.command("truth-table")
def truth_table(
    ctrl: List[float] = typer.Option(
        [],
        "--ctrl",
        help="Control powers in W (repeat: --ctrl 0 --ctrl 0.001)",
    ),
    out: Path = typer.Option(Path("truth_table.csv"), "--out", help="Output CSV"),
    threshold: str = typer.Option(
        "hard", "--threshold", help="Thresholding mode: 'hard' or 'soft'"
    ),
    beta: float = typer.Option(25.0, "--beta", help="Sigmoid slope (soft mode)"),
    xpm_mode: str = typer.Option(
        "linear", "--xpm-mode", help="XPM model: 'linear' (default) or 'physics'"
    ),
    n2: Optional[float] = typer.Option(
        None, "--n2", help="Kerr coefficient n2 (m^2/W) for physics XPM mode"
    ),
    a_eff: float = typer.Option(
        0.6e-12, "--a-eff", help="Effective mode area A_eff (m^2) for physics mode"
    ),
    n_eff: float = typer.Option(3.4, "--n-eff", help="Effective index n_eff"),
    g_geom: float = typer.Option(1.0, "--g-geom", help="Geometry scaling g_geom"),
) -> None:
    """
    Compute a truth table for control powers and write CSV.
    Column names: P_ctrl_W, T_through, T_drop, etc.
    """
    powers = [float(p) for p in (ctrl if ctrl else [0.0, 0.001, 0.002])]
    dev = PhotonicMolecule(xpm_mode=xpm_mode, n2=n2, A_eff=a_eff, n_eff=n_eff, g_geom=g_geom)
    omega = dev.omega0

    rows = []
    for P in powers:
        resp = dev.steady_state_response(omega, P)
        rows.append({"P_ctrl_W": P, **resp})
    df = pd.DataFrame(rows)
    thr = 0.5
    if (threshold or "").lower() == "soft":
        df["logic_out_soft"] = df["T_through"].apply(
            lambda v: float(soft_logic(float(v), thr, beta))
        )
    else:
        df["logic_out"] = (df["T_through"] > thr).astype(int)

    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    typer.echo(f"Wrote {out}")


@app.command("cascade")
def cascade(
    stages: int = typer.Option(2, "--stages", help="Number of cascaded stages"),
    threshold: str = typer.Option(
        "hard", "--threshold", help="Thresholding mode: 'hard' or 'soft'"
    ),
    beta: float = typer.Option(25.0, "--beta", help="Sigmoid slope (soft mode)"),
    xpm_mode: str = typer.Option(
        "linear", "--xpm-mode", help="XPM model: 'linear' (default) or 'physics'"
    ),
    n2: Optional[float] = typer.Option(
        None, "--n2", help="Kerr coefficient n2 (m^2/W) for physics XPM mode"
    ),
    a_eff: float = typer.Option(
        0.6e-12, "--a-eff", help="Effective mode area A_eff (m^2) for physics mode"
    ),
    n_eff: float = typer.Option(3.4, "--n-eff", help="Effective index n_eff"),
    g_geom: float = typer.Option(1.0, "--g-geom", help="Geometry scaling g_geom"),
) -> None:
    """
    Simulate simple cascade outputs and print JSON.
    """
    dev = PhotonicMolecule(xpm_mode=xpm_mode, n2=n2, A_eff=a_eff, n_eff=n_eff, g_geom=g_geom)
    ctl = ExperimentController(dev)
    res = ctl.test_cascade(n_stages=stages, threshold_mode=threshold, beta=beta)
    typer.echo(json.dumps(res, indent=2))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
