import argparse, json
import numpy as np
from .controller import PhotonicMolecule, ExperimentController, generate_design_report, TWOPI

def cmd_characterize(args):
    dev = PhotonicMolecule()
    ctl = ExperimentController(dev)
    results = ctl.run_full_characterization()
    ctl.results['cascade'] = ctl.test_cascade(n_stages=args.stages)
    rep = generate_design_report(dev, ctl.results, filename=args.report)
    print(json.dumps(rep, indent=2))
    print(f"Saved report to {args.report}")

def cmd_truth_table(args):
    dev = PhotonicMolecule()
    ctl = ExperimentController(dev)
    omega = dev.omega0
    powers = [float(p) for p in args.ctrl]
    import pandas as pd
    rows = []
    for P in powers:
        resp = dev.steady_state_response(omega, P)
        rows.append({'P_ctrl_W': P, **resp})
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"Wrote {args.out}")

def cmd_cascade(args):
    dev = PhotonicMolecule()
    ctl = ExperimentController(dev)
    res = ctl.test_cascade(n_stages=args.stages)
    print(json.dumps(res, indent=2))

def main():
    p = argparse.ArgumentParser(prog="plogic", description="Programmable Photonic Logic CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_char = sub.add_parser("characterize", help="Run default characterization and save report JSON")
    p_char.add_argument("--stages", type=int, default=2, help="Cascade stages for the demo")
    p_char.add_argument("--report", type=str, default="photonic_logic_report.json")
    p_char.set_defaults(func=cmd_characterize)

    p_tt = sub.add_parser("truth-table", help="Compute truth table for control powers")
    p_tt.add_argument("--ctrl", nargs="+", default=["0", "0.001", "0.002"], help="Control powers in W")
    p_tt.add_argument("--out", type=str, default="truth_table.csv")
    p_tt.set_defaults(func=cmd_truth_table)

    p_cas = sub.add_parser("cascade", help="Simulate simple cascade outputs")
    p_cas.add_argument("--stages", type=int, default=2)
    p_cas.set_defaults(func=cmd_cascade)

    args = p.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
