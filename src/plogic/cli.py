from __future__ import annotations

import importlib.metadata
import json
import math
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer

from .controller import (
    ExperimentController,
    PhotonicMolecule,
    generate_design_report,
)
from .materials.hybrid import HybridPlatform
from .materials.platforms import PlatformDB
from .utils.switching import sigmoid
from .optimization.photonic_objectives import run_photonic_optimization, create_photonic_optimizer

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


@app.command("demo")
def demo(
    gate: str = typer.Option("XOR", "--gate", help="Logic gate type (AND, OR, XOR, NAND, NOR, XNOR)"),
    platform: str = typer.Option("Si", "--platform", help="Material platform (Si, SiN, AlGaAs)"),
    threshold: str = typer.Option("hard", "--threshold", help="Threshold type (hard, soft)"),
    output: str = typer.Option("json", "--output", help="Output format (json, truth-table, csv)"),
    p_high_mw: float = typer.Option(1.0, "--P-high-mW", help="High control power in mW"),
    pulse_ns: float = typer.Option(1.0, "--pulse-ns", help="Pulse duration in ns"),
    coupling_eta: float = typer.Option(0.9, "--coupling-eta", help="Coupling efficiency"),
    link_length_um: float = typer.Option(50.0, "--link-length-um", help="Link length in um"),
) -> None:
    """
    Demonstrate logic gate operation with specified parameters.
    """
    # Load platform and configure device with platform-specific parameters
    db = PlatformDB()
    platform_obj = db.get(platform)
    
    # Create device with platform-specific parameters
    dev = PhotonicMolecule(
        n2=platform_obj.nonlinear.n2_m2_per_W,
        xpm_mode="physics" if platform == "AlGaAs" else "linear"
    )
    
    # Apply platform-specific power scaling for proper XPM effect
    # Scale power inversely with n2 to maintain constant XPM effect across platforms
    n2_reference = 1.5e-17  # AlGaAs baseline
    n2_actual = platform_obj.nonlinear.n2_m2_per_W
    power_scale = n2_reference / n2_actual if n2_actual != 0 else 1.0
    
    # Configure control parameters with platform scaling
    P_ctrl_low = 0.0
    P_ctrl_high = (p_high_mw * 1e-3) * power_scale  # Scale for platform
    
    # Use the same logic implementation as cascade command for consistency
    ctl = ExperimentController(dev)
    base_P_ctrl_W = P_ctrl_high
    pulse_duration_s = pulse_ns * 1e-9
    
    # Run the cascade simulation for just the specified gate
    cascade_res = ctl.test_cascade(
        n_stages=1,  # Single stage for demo
        base_P_ctrl_W=base_P_ctrl_W,
        pulse_duration_s=pulse_duration_s,
        threshold_mode=threshold
    )
    
    # Extract results for the specified gate
    gate_upper = gate.upper()
    if gate_upper not in cascade_res:
        # Fallback to XOR if gate not found
        gate_upper = "XOR"
    
    gate_results = cascade_res[gate_upper]
    truth_table = []
    
    # Convert cascade results to demo format
    inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]
    for i, (a, b) in enumerate(inputs):
        detail = gate_results["details"][i]
        
        if threshold == "soft":
            output_val = gate_results["logic_out_soft"][i] if "logic_out_soft" in gate_results else gate_results["outputs"][i]
        else:
            output_val = gate_results["logic_out"][i] if "logic_out" in gate_results else (1 if gate_results["outputs"][i] > 0.5 else 0)
        
        truth_table.append({
            "A": a,
            "B": b,
            "P_ctrl_mW": detail["P_ctrl"] * 1e3,
            "T_through": detail["signal"],  # Use signal from cascade simulation
            "Output": output_val,
            "Gate": gate_upper,
            "Platform": platform,
            "Threshold": threshold
        })
    
    # Output results
    if output == "truth-table" or output == "csv":
        df = pd.DataFrame(truth_table)
        if output == "csv":
            df.to_csv("truth_table.csv", index=False)
            typer.echo("Saved truth table to truth_table.csv")
        else:
            typer.echo(df.to_string(index=False))
    else:
        result = {
            "gate": gate_upper,
            "platform": platform,
            "threshold": threshold,
            "parameters": {
                "P_high_mW": p_high_mw,
                "pulse_ns": pulse_ns,
                "coupling_eta": coupling_eta,
                "link_length_um": link_length_um
            },
            "truth_table": truth_table
        }
        typer.echo(json.dumps(result, indent=2))


@app.command("cascade")
def cascade(
    stages: int = typer.Option(2, "--stages", help="Number of cascaded stages"),
    platform: Optional[str] = typer.Option(None, "--platform", help="Material platform (Si, SiN, AlGaAs)"),
    fanout: int = typer.Option(1, "--fanout", help="Fanout degree for parallelism"),
    split_loss_db: float = typer.Option(0.5, "--split-loss-db", help="Splitting loss in dB"),
    hybrid: bool = typer.Option(False, "--hybrid", help="Use hybrid platform (AlGaAs/SiN)"),
    routing_fraction: float = typer.Option(0.5, "--routing-fraction", help="Fraction of routing vs logic (hybrid only)"),
    report: Optional[str] = typer.Option(None, "--report", help="Report type (power, timing, all)"),
    p_high_mw: Optional[float] = typer.Option(None, "--P-high-mW", help="High control power in mW"),
    pulse_ns: Optional[float] = typer.Option(None, "--pulse-ns", help="Pulse duration in ns"),
    coupling_eta: Optional[float] = typer.Option(None, "--coupling-eta", help="Coupling efficiency"),
    link_length_um: Optional[float] = typer.Option(None, "--link-length-um", help="Link length in um"),
    include_2pa: bool = typer.Option(False, "--include-2pa", help="Include two-photon absorption"),
    auto_timing: bool = typer.Option(False, "--auto-timing", help="Auto-optimize timing"),
    show_resolved: bool = typer.Option(False, "--show-resolved", help="Show resolved parameters"),
    n2: Optional[float] = typer.Option(None, "--n2", help="Override nonlinear index"),
    q_factor: Optional[float] = typer.Option(None, "--q-factor", help="Override Q-factor"),
) -> None:
    """
    Simulate cascade with advanced options including fanout and hybrid platforms.
    """
    # Load platform and configure device
    platform_name = platform or "AlGaAs"
    db = PlatformDB()
    platform_obj = db.get(platform_name)
    
    # Apply platform-specific optimized defaults if not explicitly provided
    platform_defaults = {
        "AlGaAs": {
            "p_high_mw": 0.06,
            "pulse_ns": 1.4,
            "coupling_eta": 0.98,
            "link_length_um": 60.0
        },
        "Si": {
            "p_high_mw": 0.13,  # 2.2x baseline
            "pulse_ns": 0.1,    # Sub-100ps switching
            "coupling_eta": 0.9,
            "link_length_um": 50.0
        },
        "SiN": {
            "p_high_mw": 2.5,   # 42x baseline for ultra-stable
            "pulse_ns": 1.0,
            "coupling_eta": 0.9,
            "link_length_um": 20.0
        }
    }
    
    # Use platform defaults if parameters not explicitly provided
    defaults = platform_defaults.get(platform_name, platform_defaults["AlGaAs"])
    p_high_mw = p_high_mw if p_high_mw is not None else defaults["p_high_mw"]
    pulse_ns = pulse_ns if pulse_ns is not None else defaults["pulse_ns"]
    coupling_eta = coupling_eta if coupling_eta is not None else defaults["coupling_eta"]
    link_length_um = link_length_um if link_length_um is not None else defaults["link_length_um"]
    
    effective_n2 = platform_obj.nonlinear.n2_m2_per_W
    xpm_mode = "physics" if platform_name == "AlGaAs" else "linear"
    
    if hybrid:
        hybrid_platform = HybridPlatform(
            logic_material=platform_name,
            routing_material="SiN",
            routing_fraction=routing_fraction
        )
        eff_params = hybrid_platform.get_effective_parameters()
        effective_n2 = eff_params["effective_n2"]
        effective_index = eff_params["effective_index"]
        # Assuming A_eff weighted average; adjust if needed
        logic_Aeff = platform_obj.nonlinear.Aeff_um2_default * 1e-12
        routing_platform = db.get("SiN")
        routing_Aeff = routing_platform.nonlinear.Aeff_um2_default * 1e-12
        effective_Aeff = (1 - routing_fraction) * logic_Aeff + routing_fraction * routing_Aeff
    
    dev = PhotonicMolecule(
        n2=effective_n2,
        xpm_mode=xpm_mode
    )
    if hybrid:
        dev.n_eff = effective_index
        dev.A_eff = effective_Aeff
    
    # Configure platform
    platform_name = "Default"
    if hybrid:
        platform_name = "Hybrid-AlGaAs/SiN"
    elif platform:
        platform_name = platform
    
    # Apply overrides
    if n2 is not None:
        dev.n2 = n2
    if q_factor is not None:
        # Set Q factor by adjusting kappa_A (since Q = omega0 / kappa_A)
        dev.kappa_A = dev.omega0 / q_factor
        dev.kappa_B = dev.kappa_A  # Keep symmetric for simplicity
    
    # Calculate parameters
    base_P_ctrl_W = p_high_mw * 1e-3
    pulse_duration_s = pulse_ns * 1e-9
    
    # Run cascade simulation
    ctl = ExperimentController(dev)
    res = ctl.test_cascade(
        n_stages=stages,
        base_P_ctrl_W=base_P_ctrl_W,
        pulse_duration_s=pulse_duration_s
    )
    
    # Add platform and configuration info
    for gate_type in res:
        res[gate_type]["platform"] = platform_name
        res[gate_type]["fanout"] = fanout
        res[gate_type]["split_loss_db"] = split_loss_db
        res[gate_type]["effective_cascade_depth"] = stages
        
        # Calculate fanout-adjusted metrics
        if fanout > 1:
            # Fanout reduces effective depth but increases total energy
            effective_depth = max(1, int(stages / math.sqrt(fanout)))  # Use sqrt scaling as documented
            split_efficiency = 10 ** (-split_loss_db / 10)
            
            res[gate_type]["effective_cascade_depth"] = effective_depth
            res[gate_type]["split_efficiency"] = split_efficiency
            # Total energy increases with fanout (parallel operations)
            base_energy = res[gate_type].get("base_energy_fJ", base_P_ctrl_W * pulse_duration_s * 1e15)
            res[gate_type]["fanout_adjusted_energy_fJ"] = base_energy * fanout
        
        # Add hybrid platform info
        if hybrid:
            res[gate_type]["routing_fraction"] = routing_fraction
            res[gate_type]["logic_fraction"] = 1 - routing_fraction
    
    # Add power report if requested
    if report == "power":
        # Use values from simulation for consistency
        # Assuming all gates have same metrics
        sample_gate = "AND"
        power_summary = {
            "total_power_mW": res[sample_gate]["effective_P_ctrl_mW"],
            "pulse_energy_fJ": res[sample_gate]["base_energy_fJ"],
            "power_scale_factor": res[sample_gate]["power_scale_factor"],
            "platform": platform_name,
            "coupling_efficiency": coupling_eta,
            "link_length_um": link_length_um
        }
        res["power_report"] = power_summary
    
    # Show resolved parameters if requested
    if show_resolved:
        # Calculate Q factor from omega0 and kappa_A
        q_factor = dev.omega0 / dev.kappa_A if dev.kappa_A != 0 else 0
        resolved_params = {
            "n2": dev.n2,
            "Q_factor": q_factor,
            "omega0": dev.omega0,
            "alpha": getattr(dev, 'alpha', 0.0),  # alpha might not exist
            "platform": platform_name,
            "fanout": fanout,
            "stages": stages
        }
        res["resolved_parameters"] = resolved_params
    
    typer.echo(json.dumps(res, indent=2))


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


@app.command("sweep")
def sweep(
    platforms: List[str] = typer.Option([], "--platforms", help="Material platforms to sweep (Si, SiN, AlGaAs)"),
    fanout: List[int] = typer.Option([], "--fanout", help="Fanout values to sweep"),
    split_loss_db: List[float] = typer.Option([], "--split-loss-db", help="Split loss values to sweep (dB)"),
    routing_fraction: List[float] = typer.Option([], "--routing-fraction", help="Routing fraction values to sweep"),
    p_high_mw: List[float] = typer.Option([], "--P-high-mW", help="High power values to sweep (mW)"),
    pulse_ns: List[float] = typer.Option([], "--pulse-ns", help="Pulse duration values to sweep (ns)"),
    stages: List[int] = typer.Option([2], "--stages", help="Stage count values to sweep"),
    csv: Optional[Path] = typer.Option(None, "--csv", help="Output CSV file path"),
    gate: str = typer.Option("XOR", "--gate", help="Logic gate to analyze"),
) -> None:
    """
    Perform parameter sweeps and generate comparison data.
    """
    import itertools
    
    # Set defaults if no values provided
    if not platforms:
        platforms = ["Si"]
    if not fanout:
        fanout = [1]
    if not split_loss_db:
        split_loss_db = [0.5]
    if not routing_fraction:
        routing_fraction = [0.5]
    if not p_high_mw:
        p_high_mw = [1.0]
    if not pulse_ns:
        pulse_ns = [1.0]
    
    results = []
    
    # Generate all combinations
    for platform, fo, split_loss, routing_frac, p_high, pulse_dur, stage_count in itertools.product(
        platforms, fanout, split_loss_db, routing_fraction, p_high_mw, pulse_ns, stages
    ):
        # Create device
        dev = PhotonicMolecule()
        
        # Configure platform
        platform_name = platform
        hybrid = False
        
        # Check if we should use hybrid mode (when routing_fraction != 0.5 and platform is not explicitly set)
        if routing_frac != 0.5 and len(platforms) == 1 and platforms[0] in ["Si", "SiN", "AlGaAs"]:
            hybrid = True
            platform_name = f"Hybrid-{platform}/SiN"
        
        # Run cascade simulation
        ctl = ExperimentController(dev)
        res = ctl.test_cascade(n_stages=stage_count)
        
        # Extract results for the specified gate
        gate_upper = gate.upper()
        if gate_upper in res:
            gate_res = res[gate_upper]
            
            # Calculate metrics
            effective_depth = max(1, stage_count // fo) if fo > 1 else stage_count
            split_efficiency = 10 ** (-split_loss / 10) if fo > 1 else 1.0
            base_energy = gate_res.get("base_energy_fJ", 10000)
            adjusted_energy = base_energy * split_efficiency
            
            result_row = {
                "platform": platform_name,
                "fanout": fo,
                "split_loss_db": split_loss,
                "routing_fraction": routing_frac if hybrid else None,
                "P_high_mW": p_high,
                "pulse_ns": pulse_dur,
                "stages": stage_count,
                "effective_depth": effective_depth,
                "split_efficiency": split_efficiency,
                "min_contrast_dB": gate_res.get("min_contrast_dB", 0),
                "base_energy_fJ": base_energy,
                "adjusted_energy_fJ": adjusted_energy,
                "pulse_energy_fJ": p_high * pulse_dur,
                "gate": gate_upper,
                "hybrid": hybrid
            }
            
            results.append(result_row)
    
    # Output results
    if csv:
        df = pd.DataFrame(results)
        csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv, index=False)
        typer.echo(f"Saved sweep results to {csv}")
        typer.echo(f"Generated {len(results)} parameter combinations")
    else:
        typer.echo(json.dumps(results, indent=2))


@app.command("optimize")
def optimize(
    objective: str = typer.Option("multi", "--objective", help="Optimization objective (energy, cascade, thermal, multi)"),
    iterations: int = typer.Option(100, "--iterations", help="Number of DANTE optimization iterations"),
    initial_samples: int = typer.Option(30, "--initial-samples", help="Number of initial random samples"),
    samples_per_iter: int = typer.Option(5, "--samples-per-iter", help="Number of samples per DANTE iteration"),
    energy_weight: float = typer.Option(0.4, "--energy-weight", help="Weight for energy objective (multi-objective only)"),
    cascade_weight: float = typer.Option(0.3, "--cascade-weight", help="Weight for cascade objective (multi-objective only)"),
    thermal_weight: float = typer.Option(0.2, "--thermal-weight", help="Weight for thermal objective (multi-objective only)"),
    fabrication_weight: float = typer.Option(0.1, "--fabrication-weight", help="Weight for fabrication objective (multi-objective only)"),
    dims: int = typer.Option(12, "--dims", help="Number of optimization dimensions"),
    output: Optional[Path] = typer.Option(None, "--output", help="Output file for optimization results"),
    verbose: bool = typer.Option(False, "--verbose", help="Verbose optimization output"),
) -> None:
    """
    Run DANTE-powered AI optimization for photonic logic circuits.
    
    Automatically discovers optimal configurations using deep active learning.
    Supports single and multi-objective optimization across energy, cascade depth, thermal safety, and fabrication feasibility.
    """
    try:
        typer.echo(f"🚀 Starting DANTE optimization for photonic logic circuits...")
        typer.echo(f"Objective: {objective}")
        typer.echo(f"Iterations: {iterations}, Initial samples: {initial_samples}")
        typer.echo(f"Dimensions: {dims}")
        
        if objective == "multi":
            typer.echo(f"Multi-objective weights: Energy={energy_weight}, Cascade={cascade_weight}, Thermal={thermal_weight}, Fab={fabrication_weight}")
        
        # Run DANTE optimization
        results = run_photonic_optimization(
            objective_type=objective,
            num_initial_samples=initial_samples,
            num_acquisitions=iterations,
            samples_per_acquisition=samples_per_iter,
            energy_weight=energy_weight,
            cascade_weight=cascade_weight,
            thermal_weight=thermal_weight,
            fabrication_weight=fabrication_weight,
            dims=dims
        )
        
        # Display results
        typer.echo(f"\n✅ Optimization completed!")
        typer.echo(f"Total evaluations: {results['total_evaluations']}")
        typer.echo(f"Best score: {results['best_score']:.4f}")
        
        # Decode best parameters
        best_params = results['best_parameters']
        platforms = ["AlGaAs", "Si", "SiN"]
        platform_idx = int(np.clip(best_params[0], 0, 2))
        
        typer.echo(f"\n🎯 Best Configuration:")
        typer.echo(f"Platform: {platforms[platform_idx]}")
        typer.echo(f"Power: {best_params[1]:.3f} mW")
        typer.echo(f"Pulse: {best_params[2]:.3f} ns")
        typer.echo(f"Coupling: {best_params[3]:.3f}")
        typer.echo(f"Link length: {best_params[4]:.1f} μm")
        typer.echo(f"Fanout: {int(best_params[5])}")
        typer.echo(f"Split loss: {best_params[6]:.2f} dB")
        typer.echo(f"Stages: {int(best_params[7])}")
        
        if dims >= 12 and len(best_params) >= 12:
            typer.echo(f"Hybrid: {'Yes' if best_params[8] > 0.5 else 'No'}")
            if best_params[8] > 0.5:
                typer.echo(f"Routing fraction: {best_params[9]:.2f}")
        
        # Save results if requested
        if output:
            output.parent.mkdir(parents=True, exist_ok=True)
            with open(output, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            typer.echo(f"\n💾 Results saved to {output}")
        
        # Show optimization history if verbose
        if verbose:
            typer.echo(f"\n📈 Optimization History:")
            for i, sol in enumerate(results['optimization_history'][-5:]):  # Last 5 iterations
                typer.echo(f"Iter {sol['iteration']}: Score={sol['best_score']:.4f}, Evals={sol['num_evaluations']}")
        
    except ImportError as e:
        typer.echo(f"❌ Error: DANTE not properly installed. Please install DANTE first:")
        typer.echo(f"pip install git+https://github.com/Bop2000/DANTE.git")
        typer.echo(f"Error details: {e}")
    except Exception as e:
        typer.echo(f"❌ Optimization failed: {e}")
        if verbose:
            import traceback
            typer.echo(traceback.format_exc())


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
