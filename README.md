# Programmable Photonic Logic (v2.2)

A practical, end-to-end toolkit for programmable photonic logic: simulate and characterize coupled-cavity "photonic molecule" gates, run automated XPM-based switching experiments, and scale from room-temperature pJ-class devices to quantum upgrades (QD/PhC & Rydberg-EIT). Includes a Typer CLI, unit tests, bring-up playbooks, and hardware control hooks.

## Quickstart

```bash
# Create and activate a virtual env (recommended)
python -m venv .venv && source .venv/bin/activate

# Install from source (editable)
pip install -e .

# Run a default characterization and generate report + plots
plogic characterize

# Generate a truth table CSV for a few control powers
plogic truth-table --ctrl 0 0.0005 0.001 0.002

# Simulate a simple cascade (2 stages) and print results
plogic cascade --stages 2
```

See docs/guide_v2.2.md for the full buildable manual (theoretical foundations, fabrication cribsheet, thermal playbook, Python APIs, and quantum upgrade path).

> For a lightweight one-file harness, see examples/photonic_logic_test_v11.py.

## New in v2.2 (P0 + P1)

These features are implemented with backward-compatible defaults. Existing scripts continue to work without changes.

- Soft thresholding utilities and cascade integration
  - Smooth, β-tunable logic mapping to reduce brittle toggling in cascades.
  - Utilities: sigmoid, softplus, soft_logic, hard_logic (see plogic.utils).
  - CLI toggles: --threshold {hard,soft}, --beta FLOAT.

- Physics-based XPM detuning path
  - Kerr model: Δn = n2 · (P_ctrl / A_eff), Δω ≈ −(ω0/n_eff) · g_geom · Δn.
  - Select between legacy linear shortcut and physics mode.
  - CLI toggles: --xpm-mode {linear,physics}, plus --n2, --a-eff, --n-eff, --g-geom.

## CLI usage

All commands support optional thresholding and XPM modeling flags.

- characterize
  - Run a default characterization; include cascade results using the selected thresholding mode.
  - Example (soft threshold + physics XPM):
    ```
    plogic characterize --threshold soft --beta 30 --xpm-mode physics --n2 1e-17
    ```
- truth-table
  - Generate CSV across control powers.
  - Soft threshold adds a probability-like column logic_out_soft in (0,1); hard adds binary logic_out.
  - Example:
    ```
    plogic truth-table --ctrl 0 --ctrl 0.001 --threshold soft --beta 25
    ```
- cascade
  - Simulate simple cascade and return JSON.
  - Example (physics-based XPM):
    ```
    plogic cascade --threshold soft --beta 30 --xpm-mode physics --n2 1e-17
    ```

Defaults:
- Thresholding: hard (0/1 at 0.5)
- XPM mode: linear (legacy g_XPM * P_ctrl)
- Physics params (if used): n2 (m^2/W), A_eff (m^2), n_eff (unitless), g_geom (unitless)

## Running without installing the package

If you prefer not to install in editable mode, you can run the CLI as a module.

- macOS/Linux (bash/zsh):
  ```
  export PYTHONPATH=src
  python -m plogic.cli --help
  ```
- Windows PowerShell:
  ```
  $env:PYTHONPATH="src"; py -m plogic.cli --help
  ```

Troubleshooting:
- If you see ImportError: attempted relative import with no known parent package, either:
  - Install the package: pip install -e . (recommended), then use the plogic entrypoint, or
  - Use module invocation with PYTHONPATH as shown above.

## Python versions and dev setup

- Requires Python ≥ 3.10 (tested 3.10–3.12).
- Dev extras (linting/formatting/testing): see pyproject.toml optional dependencies (dev).
  ```
  pip install -e .[dev]
  pytest -q
  ```
- Code style: black, ruff (configured in pyproject.toml).

## Notes and roadmap

- The cascade logic labels (AND/OR/XOR) currently serve as placeholders using a common transfer pathway; logic-specific transfer shaping can be extended in future work.
- Upcoming milestones:
  - P2: Unified temperature model (κ_sc(T_op)+κ_ph(T_op)) integrated into losses/dispersion.
  - P3: Π wrapper and optimizer with a pi-eval CLI for workload analysis.
  - P4: Calibration hooks and persistence to YAML/JSON.
  - P5: Docs/examples: Π–P_ctrl maps and T_op sensitivity.

## Citation

If this toolkit helps your work, please cite the project using CITATION.cff.
