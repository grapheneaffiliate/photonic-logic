from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import typer

from .controller import (
    PhotonicMolecule,
    ExperimentController,
    generate_design_report,
)

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Programmable Photonic Logic CLI",
)


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
    Mirrors the current argparse CLI behavior.
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
    ctrl: Optional[List[float]] = typer.Option(
        None,
        "--ctrl",
        help="Control powers in W. Provide multiple as repeated options: --ctrl 0 --ctrl 0.001",
        multiple=True,
    ),
    out: Path = typer.Option(Path("truth_table.csv"), "--out", help="Output CSV"),
) -> None:
    """
    Compute a truth table for control powers and write CSV.
    Column names match the current CLI (P_ctrl_W, T_through, T_drop, ...).
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
    app()


if __name__ == "__main__":
    main()
