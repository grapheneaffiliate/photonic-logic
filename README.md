# Programmable Photonic Logic (v2.2)

Turn-key framework for designing, characterizing, and operating *programmable light-light logic* using coupled resonators (photonic molecules).

## Quickstart

```bash
# Create and activate a virtual env (recommended)
python -m venv .venv && source .venv/bin/activate

# Install from source
pip install -e .

# Run a default characterization and generate report + plots
plogic characterize

# Generate a truth table CSV for a few control powers
plogic truth-table --ctrl 0 0.0005 0.001 0.002

# Simulate a simple cascade (2 stages) and print results
plogic cascade --stages 2
```

See **docs/guide_v2.2.md** for the full buildable manual (theoretical foundations, fabrication cribsheet, thermal playbook, Python APIs, and quantum upgrade path).

> For a lightweight one-file harness, see **examples/photonic_logic_test_v11.py**.
