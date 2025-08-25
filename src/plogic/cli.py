from __future__ import annotations

import json
import importlib.metadata
from pathlib import Path
from typing import List, Optional

import pandas as pd
import typer

from .controller import (
    PhotonicMolecule,
    ExperimentController,
    generate_design_report,
)

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


def main() -> None:
    # Keep console_script target compatible with pyproject: plogic = "plogic.cli:main"
    # Back-compat: rewrite '--ctrl 0 0.001' into '--ctrl 0 --ctrl 0.001' for Typer parsing.
    import sys
    argv = sys.argv[1:]
    if "--ctrl" in argv:
        new_argv: list[str] = []
        i = 0
        while i < len(argv):
            if argv[i] == "--ctrl":
                i += 1
                # Capture subsequent non-option tokens as values
                while i < len(argv) and not argv[i].startswith("-"):
                    new_argv.extend(["--ctrl", argv[i]])
                    i += 1
                continue
            new_argv.append(argv[i])
            i += 1
        sys.argv = [sys.argv[0]] + new_argv
    app()


if __name__ == "__main__":
    main()
