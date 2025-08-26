# Programmable Photonic Logic (v2.2)

A practical, end-to-end toolkit for programmable photonic logic: simulate and characterize coupled-cavity “photonic molecule” gates, run automated XPM-based switching experiments, and scale from room‑temperature pJ‑class devices to quantum upgrades (QD/PhC & Rydberg‑EIT). Includes a Typer CLI, unit tests, bring-up playbooks, and hardware control hooks.

## What’s new in v2.2 (P0 + P1)

These features are implemented with backward‑compatible defaults. Existing scripts continue to work without changes.

- Soft thresholding utilities and cascade integration
  - Smooth, β‑tunable logic mapping to reduce brittle toggling in cascades.
  - Utilities: `sigmoid`, `softplus`, `soft_logic`, `hard_logic` (see `plogic.utils`).
  - CLI toggles: `--threshold {hard,soft}`, `--beta FLOAT`.

- Physics‑based XPM detuning path
  - Kerr model: Δn = n2 · (P_ctrl / A_eff), Δω ≈ −(ω0/n_eff) · g_geom · Δn.
  - Select between legacy linear shortcut and physics mode.
  - CLI toggles: `--xpm-mode {linear,physics}`, plus `--n2`, `--a-eff`, `--n-eff`, `--g-geom`.

## Getting Started

Pick one of the following setups.

- pip (editable install)
  ```bash
  git clone https://github.com/grapheneaffiliate/photonic-logic.git
  cd photonic-logic
  python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
  pip install -r requirements.txt
  pip install -r requirements-dev.txt  # optional: linting/tests/docs
  pip install -e .
  ```

- Conda (reproducible environment)
  ```bash
  git clone https://github.com/grapheneaffiliate/photonic-logic.git
  cd photonic-logic
  conda env create -f environment.yml
  conda activate photonic-logic
  pip install -e .[dev]
  ```

## Quick CLI usage

- Run a default characterization and print a JSON report
  ```bash
  plogic characterize
  ```

- Generate a truth table CSV for a few control powers
  ```bash
  plogic truth-table --ctrl 0 0.0005 0.001 0.002 --out data/reports/truth_table.csv
  ```

- Simulate a cascade with soft thresholding and physics‑based XPM
  ```bash
  plogic cascade --threshold soft --beta 30 --xpm-mode physics --n2 1e-17
  ```

If you prefer not to install in editable mode, you can run the CLI as a module:

- macOS/Linux
  ```bash
  export PYTHONPATH=src
  python -m plogic.cli --help
  ```

- Windows PowerShell
  ```powershell
  $env:PYTHONPATH="src"; py -m plogic.cli --help
  ```

Troubleshooting:
- If you see “attempted relative import with no known parent package”, either:
  - Install the package: `pip install -e .` (recommended), or
  - Use module invocation with `PYTHONPATH` as shown above.

## Example analysis notebook and sample data

- A small, representative dataset: `data/sample/truth_table_sample.csv`
- A minimal analysis notebook: `notebooks/Example_Analysis.ipynb`

Open the notebook to plot transmission vs control power, or regenerate a CSV using:
```bash
plogic truth-table --ctrl 0 --ctrl 0.001 --ctrl 0.002 --out data/sample/truth_table_generated.csv
```

## Repo structure (high level)

- `src/plogic/` – core Python package (device model, controllers, CLI)
- `tests/` – unit tests (run with `pytest -q`)
- `docs/` – guides and references
- `examples/` – quick experimentation scripts
- `data/` – sample and report outputs
- `design/`, `hardware/` – layout and hardware stubs/placeholders

## Development

- Lint/format/test locally:
  ```bash
  ruff check .
  black --check .
  pytest -q
  ```
- Pre-commit hooks:
  ```bash
  pre-commit install
  pre-commit run --all-files
  ```

## Citation

If this toolkit helps your work, please cite the project. A quick BibTeX entry:

```bibtex
@software{photonic_logic_v2_2,
  author  = {Open Photonics Lab},
  title   = {Programmable Photonic Logic (v2.2)},
  year    = {2025},
  url     = {https://github.com/grapheneaffiliate/photonic-logic},
  note    = {Soft switching, physics-based XPM, CLI tooling, tests}
}
```

For a formal citation, see `CITATION.cff` at the repository root.

## License

MIT License. See `LICENSE`.
