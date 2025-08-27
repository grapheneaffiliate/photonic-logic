from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path
from typing import List, Optional

import pandas as pd
import typer

from .analysis import PowerInputs, compute_power_report
from .controller import (
    ExperimentController,
    PhotonicMolecule,
    generate_design_report,
)
from .materials import PlatformDB
from .utils import soft_logic
from .utils.io import save_csv, save_json

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
        "physics", "--xpm-mode", help="XPM model: 'linear' or 'physics' (default)"
    ),
    # Material platform selection
    platform: Optional[str] = typer.Option(
        None, "--platform", help="Material platform: Si, SiN, or AlGaAs"
    ),
    # Physics parameter overrides (backward compatible)
    n2: Optional[float] = typer.Option(None, "--n2", help="Kerr coefficient n2 (m^2/W) override"),
    a_eff: float = typer.Option(
        0.6e-12, "--a-eff", help="Effective mode area A_eff (m^2) for physics mode"
    ),
    n_eff: float = typer.Option(3.4, "--n-eff", help="Effective index n_eff"),
    g_geom: float = typer.Option(1.0, "--g-geom", help="Geometry scaling g_geom"),
    lambda_nm: Optional[float] = typer.Option(
        None, "--lambda-nm", help="Operating wavelength [nm]"
    ),
    aeff_um2: Optional[float] = typer.Option(None, "--aeff-um2", help="Effective mode area [um^2]"),
    q_factor: Optional[float] = typer.Option(
        None, "--q-factor", help="Intrinsic cavity Q (override)"
    ),
    loss_dB_cm: Optional[float] = typer.Option(
        None, "--loss-dB-cm", help="Waveguide loss [dB/cm] override"
    ),
    # Power reporting
    report: str = typer.Option("none", "--report", help="Report type: 'none' or 'power'"),
    # Power report parameters
    P_high_mW: float = typer.Option(1.0, "--P-high-mW", help="Logic-1 drive power [mW]"),
    threshold_norm: float = typer.Option(0.5, "--threshold-norm", help="Normalized threshold [0..1]"),
    fanout: int = typer.Option(1, "--fanout", help="Fan-out per stage"),
    coupling_eta: float = typer.Option(0.8, "--coupling-eta", help="Coupling efficiency [0..1]"),
    link_length_um: float = typer.Option(
        50.0, "--link-length-um", help="Link length per stage [um]"
    ),
    pulse_ns: Optional[float] = typer.Option(None, "--pulse-ns", help="Pulse width [ns]"),
    bitrate_GHz: Optional[float] = typer.Option(None, "--bitrate-GHz", help="Bit rate [GHz]"),
    auto_timing: bool = typer.Option(False, "--auto-timing", help="Derive timing from Q factor"),
    include_2pa: bool = typer.Option(False, "--include-2pa", help="Include two-photon absorption"),
    extinction_target_dB: float = typer.Option(
        21.0, "--extinction-target-dB", help="Target extinction ratio [dB]"
    ),
    er_epsilon: float = typer.Option(1e-12, "--er-epsilon", help="Tolerance for ER boundary check"),
    L_eff_um: float = typer.Option(10.0, "--L-eff-um", help="Effective interaction length [um]"),
    # Output options
    save_primary: Optional[Path] = typer.Option(
        None, "--save-primary", help="Save cascade JSON to file"
    ),
    save_report: Optional[Path] = typer.Option(
        None, "--save-report", help="Save power report JSON to file"
    ),
    csv: Optional[Path] = typer.Option(None, "--csv", help="Save power report as CSV"),
    embed_report: bool = typer.Option(
        False, "--embed-report", help="Embed power report in cascade JSON"
    ),
    quiet: bool = typer.Option(False, "--quiet", help="Suppress stdout JSON"),
    show_resolved: bool = typer.Option(False, "--show-resolved", help="Show resolved parameters"),
) -> None:
    """
    Simulate cascade outputs with optional material platform and power analysis.
    """
    # Resolve platform parameters
    platform_obj = None
    if platform:
        pdb = PlatformDB()
        platform_obj = pdb.get(platform)
        if not quiet:
            typer.echo(f"[plogic] Using platform: {platform_obj.name} ({platform_obj.key})")

    # Resolve parameters with precedence: flags > platform > defaults
    wavelength_nm = lambda_nm or (platform_obj.default_wavelength_nm if platform_obj else 1550.0)
    Aeff_um2 = aeff_um2 or (platform_obj.nonlinear.Aeff_um2_default if platform_obj else 0.6)
    n2_resolved = (
        n2 if n2 is not None else (platform_obj.nonlinear.n2_m2_per_W if platform_obj else 1e-17)
    )
    loss_dB_cm_resolved = (
        loss_dB_cm
        if loss_dB_cm is not None
        else (platform_obj.fabrication.loss_dB_per_cm if platform_obj else 0.1)
    )

    # Convert units for PhotonicMolecule (expects m^2, not um^2)
    A_eff_m2 = Aeff_um2 * 1e-12

    # Validate Q factor if platform is specified
    if platform_obj and q_factor:
        warn = platform_obj.validate_reasonable_Q(q_factor)
        if warn and not quiet:
            typer.echo(f"[plogic][warn] {warn}")

    # Show resolved parameters if requested
    if show_resolved and not quiet:
        resolved_params = {
            "platform": platform,
            "wavelength_nm": wavelength_nm,
            "n2_m2_per_W": n2_resolved,
            "A_eff_m2": A_eff_m2,
            "Aeff_um2": Aeff_um2,
            "loss_dB_cm": loss_dB_cm_resolved,
            "q_factor": q_factor,
            "xpm_mode": xpm_mode,
            "threshold": threshold,
            "beta": beta,
        }
        typer.echo("[plogic] Resolved parameters:")
        typer.echo(json.dumps(resolved_params, indent=2))

    # Create device with resolved parameters
    dev = PhotonicMolecule(
        xpm_mode=xpm_mode, n2=n2_resolved, A_eff=A_eff_m2, n_eff=n_eff, g_geom=g_geom
    )
    ctl = ExperimentController(dev)

    # Run cascade simulation
    res = ctl.test_cascade(n_stages=stages, threshold_mode=threshold, beta=beta)

    # Extract measured statistics for power analysis
    if report == "power":
        # Extract ON/OFF statistics from cascade results
        min_on_global = float("inf")
        max_off_global = 0.0

        for gate_name, gate_data in res.items():
            if "details" in gate_data and "logic_out" in gate_data:
                for detail, logic_out in zip(gate_data["details"], gate_data["logic_out"]):
                    signal = detail.get("signal", 0.0)
                    if logic_out == 1:
                        min_on_global = min(min_on_global, signal)
                    else:
                        max_off_global = max(max_off_global, signal)

        if min_on_global == float("inf"):
            min_on_global = 1.0

        worst_off_norm = max_off_global / max(min_on_global, 1e-30)

        # Build power analysis inputs
        pins = PowerInputs(
            wavelength_nm=wavelength_nm,
            platform_loss_dB_cm=loss_dB_cm_resolved,
            coupling_eta=coupling_eta,
            link_length_um=link_length_um,
            fanout=fanout,
            pulse_ns=pulse_ns,
            bitrate_GHz=bitrate_GHz,
            q_factor=q_factor,
            P_high_mW=P_high_mW,
            threshold_norm=threshold_norm,
            worst_off_norm=worst_off_norm,
            extinction_target_dB=extinction_target_dB,
            er_epsilon=er_epsilon,
            n2_m2_per_W=n2_resolved,
            Aeff_um2=Aeff_um2,
            dn_dT_per_K=(platform_obj.thermal.dn_dT_per_K if platform_obj else None),
            tau_thermal_ns=(platform_obj.thermal.tau_thermal_ns if platform_obj else None),
            L_eff_um=L_eff_um,
            include_2pa=include_2pa,
            beta_2pa_m_per_W=(platform_obj.nonlinear.beta_2pa_m_per_W if platform_obj else 0.0),
            auto_timing=auto_timing or (pulse_ns is None and bitrate_GHz is None),
        )

        power_rep = compute_power_report(pins)
        power_rep.raw["stats"] = {
            "min_on_level": min_on_global,
            "max_off_level": max_off_global,
            "worst_off_norm": worst_off_norm,
            "threshold_norm": threshold_norm,
        }

    # Output logic
    if embed_report and report == "power":
        # Embed power report in cascade JSON
        res.setdefault("analysis", {})["power"] = power_rep.raw
        if not quiet:
            typer.echo(json.dumps(res, indent=2))
        if save_primary:
            save_json(res, save_primary)
        if save_report:
            save_json(power_rep.raw, save_report)
        if csv:
            save_csv(power_rep.raw, csv)
    else:
        # Separate outputs (default behavior)
        if not quiet:
            typer.echo(json.dumps(res, indent=2))
            if report == "power":
                typer.echo(json.dumps(power_rep.raw, indent=2))

        if save_primary:
            save_json(res, save_primary)
        if report == "power":
            if save_report:
                save_json(power_rep.raw, save_report)
            if csv:
                save_csv(power_rep.raw, csv)


@app.command("sweep")
def sweep(
    platforms: List[str] = typer.Option(
        ["Si", "SiN", "AlGaAs"], "--platforms", help="Material platforms to sweep"
    ),
    beta: List[float] = typer.Option([80.0], "--beta", help="Beta values to sweep"),
    threshold: str = typer.Option("hard", "--threshold", help="Thresholding mode"),
    xpm_mode: str = typer.Option("physics", "--xpm-mode", help="XPM model"),
    P_high_mW: List[float] = typer.Option(
        [0.3, 0.5, 1.0], "--P-high-mW", help="Drive powers [mW] to sweep"
    ),
    fanout: List[int] = typer.Option([1, 2], "--fanout", help="Fan-out values to sweep"),
    coupling_eta: List[float] = typer.Option(
        [0.8], "--coupling-eta", help="Coupling efficiency values"
    ),
    link_length_um: List[float] = typer.Option(
        [50.0], "--link-length-um", help="Link lengths [um]"
    ),
    # Single-value parameters
    lambda_nm: Optional[float] = typer.Option(
        None, "--lambda-nm", help="Operating wavelength [nm]"
    ),
    aeff_um2: Optional[float] = typer.Option(None, "--aeff-um2", help="Effective mode area [um^2]"),
    q_factor: Optional[float] = typer.Option(None, "--q-factor", help="Cavity Q factor"),
    bitrate_GHz: Optional[float] = typer.Option(None, "--bitrate-GHz", help="Bit rate [GHz]"),
    pulse_ns: Optional[float] = typer.Option(None, "--pulse-ns", help="Pulse width [ns]"),
    auto_timing: bool = typer.Option(False, "--auto-timing", help="Auto-derive timing from Q"),
    include_2pa: bool = typer.Option(False, "--include-2pa", help="Include two-photon absorption"),
    n2: Optional[float] = typer.Option(None, "--n2", help="Override n2 value"),
    loss_dB_cm: Optional[float] = typer.Option(None, "--loss-dB-cm", help="Override loss value"),
    # Output options
    outdir: Optional[Path] = typer.Option(
        None, "--outdir", help="Directory for per-point JSON files"
    ),
    csv: Optional[Path] = typer.Option(None, "--csv", help="Consolidated CSV output"),
    quiet: bool = typer.Option(False, "--quiet", help="Suppress progress output"),
    show_config: bool = typer.Option(
        False, "--show-config", help="Show sweep configuration and exit"
    ),
    # Parallel processing
    parallel: bool = typer.Option(False, "--parallel", help="Run sweep in parallel"),
    workers: int = typer.Option(0, "--workers", help="Number of workers (0=auto)"),
    timeout: Optional[float] = typer.Option(None, "--timeout", help="Timeout per point [seconds]"),
) -> None:
    """
    Run parameter sweeps across material platforms with power analysis.
    """
    import itertools

    from .materials import PlatformDB

    # Build parameter grid
    axes = {
        "platform": platforms,
        "beta": beta,
        "threshold": [threshold],
        "xpm_mode": [xpm_mode],
        "P_high_mW": P_high_mW,
        "fanout": fanout,
        "coupling_eta": coupling_eta,
        "link_length_um": link_length_um,
    }

    # Add single-value parameters if specified
    if lambda_nm is not None:
        axes["lambda_nm"] = [lambda_nm]
    if aeff_um2 is not None:
        axes["aeff_um2"] = [aeff_um2]
    if q_factor is not None:
        axes["q_factor"] = [q_factor]
    if bitrate_GHz is not None:
        axes["bitrate_GHz"] = [bitrate_GHz]
    if pulse_ns is not None:
        axes["pulse_ns"] = [pulse_ns]
    if auto_timing:
        axes["auto_timing"] = [True]
    if include_2pa:
        axes["include_2pa"] = [True]
    if n2 is not None:
        axes["n2"] = [n2]
    if loss_dB_cm is not None:
        axes["loss_dB_cm"] = [loss_dB_cm]

    # Generate all combinations
    keys = sorted(axes.keys())
    combinations = list(itertools.product(*(axes[k] for k in keys)))

    if show_config:
        config = [dict(zip(keys, combo)) for combo in combinations]
        typer.echo(json.dumps(config, indent=2))
        return

    if not quiet:
        typer.echo(f"[plogic] Running {len(combinations)} sweep points...")

    # Run sweep
    pdb = PlatformDB()
    results = []

    for idx, combo in enumerate(combinations):
        params = dict(zip(keys, combo))

        try:
            # Resolve platform parameters
            platform_obj = pdb.get(params["platform"]) if params.get("platform") else None

            # Resolve parameters with precedence
            wavelength_nm_resolved = params.get("lambda_nm") or (
                platform_obj.default_wavelength_nm if platform_obj else 1550.0
            )
            Aeff_um2_resolved = params.get("aeff_um2") or (
                platform_obj.nonlinear.Aeff_um2_default if platform_obj else 0.6
            )
            n2_resolved = (
                params.get("n2")
                if params.get("n2") is not None
                else (platform_obj.nonlinear.n2_m2_per_W if platform_obj else 1e-17)
            )
            loss_dB_cm_resolved = (
                params.get("loss_dB_cm")
                if params.get("loss_dB_cm") is not None
                else (platform_obj.fabrication.loss_dB_per_cm if platform_obj else 0.1)
            )

            # Create device and run simulation
            dev = PhotonicMolecule(
                xpm_mode=params["xpm_mode"],
                n2=n2_resolved,
                A_eff=Aeff_um2_resolved * 1e-12,
                n_eff=3.4,
                g_geom=1.0,
            )
            ctl = ExperimentController(dev)
            cascade_result = ctl.test_cascade(
                n_stages=2, threshold_mode=params["threshold"], beta=params["beta"]
            )

            # Extract measured statistics
            min_on_global = float("inf")
            max_off_global = 0.0

            for gate_name, gate_data in cascade_result.items():
                if "details" in gate_data and "logic_out" in gate_data:
                    for detail, logic_out in zip(gate_data["details"], gate_data["logic_out"]):
                        signal = detail.get("signal", 0.0)
                        if logic_out == 1:
                            min_on_global = min(min_on_global, signal)
                        else:
                            max_off_global = max(max_off_global, signal)

            if min_on_global == float("inf"):
                min_on_global = 1.0

            worst_off_norm = max_off_global / max(min_on_global, 1e-30)

            # Compute power report
            pins = PowerInputs(
                wavelength_nm=wavelength_nm_resolved,
                platform_loss_dB_cm=loss_dB_cm_resolved,
                coupling_eta=params["coupling_eta"],
                link_length_um=params["link_length_um"],
                fanout=params["fanout"],
                pulse_ns=params.get("pulse_ns"),
                bitrate_GHz=params.get("bitrate_GHz"),
                q_factor=params.get("q_factor"),
                P_high_mW=params["P_high_mW"],
                threshold_norm=0.5,
                worst_off_norm=worst_off_norm,
                extinction_target_dB=20.0,
                n2_m2_per_W=n2_resolved,
                Aeff_um2=Aeff_um2_resolved,
                dn_dT_per_K=(platform_obj.thermal.dn_dT_per_K if platform_obj else None),
                tau_thermal_ns=(platform_obj.thermal.tau_thermal_ns if platform_obj else None),
                L_eff_um=10.0,
                include_2pa=params.get("include_2pa", False),
                beta_2pa_m_per_W=(platform_obj.nonlinear.beta_2pa_m_per_W if platform_obj else 0.0),
                auto_timing=params.get("auto_timing", False)
                or (params.get("pulse_ns") is None and params.get("bitrate_GHz") is None),
            )

            power_rep = compute_power_report(pins)

            # Create artifact
            artifact = {
                "meta": {
                    "index": idx,
                    "platform": params.get("platform"),
                    "beta": params["beta"],
                    "threshold": params["threshold"],
                    "xpm_mode": params["xpm_mode"],
                    "P_high_mW": params["P_high_mW"],
                    "fanout": params["fanout"],
                    "coupling_eta": params["coupling_eta"],
                    "link_length_um": params["link_length_um"],
                    "lambda_nm": wavelength_nm_resolved,
                    "aeff_um2": Aeff_um2_resolved,
                    "q_factor": params.get("q_factor"),
                    "include_2pa": params.get("include_2pa", False),
                },
                "cascade": cascade_result,
                "power": power_rep.raw,
            }

            results.append(artifact)

            # Save per-point JSON if requested
            if outdir:
                outdir.mkdir(parents=True, exist_ok=True)
                base = f"{idx:04d}_{params.get('platform', 'raw')}_b{params['beta']}_P{params['P_high_mW']}mW_FO{params['fanout']}"
                save_json(artifact, outdir / f"{base}.json")

            # Append to CSV if requested
            if csv:
                merged = {"meta": artifact["meta"], **artifact["power"]}
                save_csv(merged, csv)

            if not quiet:
                typer.echo(
                    f"[sweep] {idx+1}/{len(combinations)} done: platform={params.get('platform')} P={params['P_high_mW']}mW β={params['beta']} fanout={params['fanout']}"
                )

        except Exception as e:
            if not quiet:
                typer.echo(f"[sweep][ERROR] {idx+1}/{len(combinations)}: {e}")

    if not quiet:
        typer.echo(f"[sweep] Complete: {len(results)} points processed")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
