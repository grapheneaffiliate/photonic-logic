from __future__ import annotations

import importlib.metadata
import json
import math
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import typer

from .controller import (
    ExperimentController,
    PhotonicMolecule,
    generate_design_report,
)
from .utils.switching import sigmoid

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
) -> None:
    """
    Run default characterization and save report JSON.
    """
    dev = PhotonicMolecule()
    ctl = ExperimentController(dev)
    ctl.run_full_characterization()
    ctl.results["cascade"] = ctl.test_cascade(n_stages=stages)
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
) -> None:
    """
    Compute a truth table for control powers and write CSV.
    Column names: P_ctrl_W, T_through, T_drop, etc.
    """
    powers = [float(p) for p in (ctrl if ctrl else [0.0, 0.001, 0.002])]
    dev = PhotonicMolecule()
    omega = dev.omega0

    rows = []
    for P in powers:
        resp = dev.steady_state_response(omega, P)
        rows.append({"P_ctrl_W": P, **resp})
    df = pd.DataFrame(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    typer.echo(f"Wrote {out}")


@app.command("cascade")
def cascade(
    stages: int = typer.Option(2, "--stages", help="Number of cascaded stages"),
) -> None:
    """
    Simulate simple cascade outputs and print JSON.
    """
    dev = PhotonicMolecule()
    ctl = ExperimentController(dev)
    res = ctl.test_cascade(n_stages=stages)
    typer.echo(json.dumps(res, indent=2))


@app.command("benchmark")
def benchmark(
    metric: str = typer.Option(
        "switching-contrast",
        "--metric",
        help="Benchmark metric: 'switching-contrast' or 'cascade-stability'",
    ),
    stages: int = typer.Option(2, "--stages", help="Stages for cascade-stability"),
) -> None:
    """
    Run lightweight benchmarks and print a small JSON result.

    - switching-contrast: approximate contrast (dB) between P_ctrl=0 and P_ctrl=1 mW at omega0
    - cascade-stability: reports min_contrast_dB from test_cascade()
    """
    dev = PhotonicMolecule()
    omega = dev.omega0

    if metric == "switching-contrast":
        r_off = dev.steady_state_response(omega, P_ctrl=0.0)
        r_on = dev.steady_state_response(omega, P_ctrl=1e-3)  # 1 mW
        t_off = max(min(float(r_off["T_through"]), 1.0), 1e-12)
        t_on = max(min(float(r_on["T_through"]), 1.0), 1e-12)
        # dB contrast between ON and OFF transmissions
        contrast_db = 10.0 * math.log10(max(t_on, t_off) / max(min(t_on, t_off), 1e-12))
        out = {"metric": metric, "contrast_dB": contrast_db}
        typer.echo(json.dumps(out, indent=2))
        return

    if metric == "cascade-stability":
        ctl = ExperimentController(dev)
        res = ctl.test_cascade(n_stages=stages)
        # Aggregate minimum across logic variants for a single scalar
        mins = [res[k]["min_contrast_dB"] for k in res]
        out = {"metric": metric, "stages": stages, "min_contrast_dB": min(mins) if mins else 0.0}
        typer.echo(json.dumps(out, indent=2))
        return

    typer.echo(json.dumps({"error": f"Unknown metric: {metric}"}, indent=2))


@app.command("visualize")
def visualize(
    mode: str = typer.Option(
        "soft-threshold", "--mode", help="Visualization mode (e.g., 'soft-threshold')"
    ),
    beta: float = typer.Option(20.0, "--beta", help="Sigmoid slope for soft threshold plot"),
    out: Path = typer.Option(Path("soft_threshold.png"), "--out", help="Output image path"),
) -> None:
    """
    Produce basic visualizations to aid intuition.
    - soft-threshold: plot y = sigmoid(x - 0.5, beta) for x in [0,1].
    """
    if mode == "soft-threshold":
        import numpy as np

        x = np.linspace(0.0, 1.0, 501)
        y = sigmoid(x - 0.5, beta)
        plt.figure(figsize=(5, 3.2))
        plt.plot(x, y, label=f"sigmoid(x-0.5, beta={beta:g})")
        plt.axvline(0.5, color="k", ls="--", alpha=0.4)
        plt.xlabel("Input (normalized)")
        plt.ylabel("Output")
        plt.title("Soft Threshold (Sigmoid)")
        plt.grid(alpha=0.3)
        plt.legend()
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(out, dpi=150)
        typer.echo(f"Wrote {out}")
        return

    typer.echo(json.dumps({"error": f"Unknown mode: {mode}"}, indent=2))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
