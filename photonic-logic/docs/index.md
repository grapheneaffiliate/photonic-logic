# Photonic Logic Documentation

Welcome to the documentation site for the Photonic Logic toolkit. This site hosts the implementation guide, developer notes, and quick-starts for simulations and characterization.

- Guide v2.2 (current): theory, device model, soft thresholding (P0), physics-based XPM (P1), and CLI usage.
- Guide v1.1 (annotated): historical/annotated reference.

Quick links:
- GitHub repository: https://github.com/grapheneaffiliate/photonic-logic
- README (on GitHub): project overview, installation, CLI examples
- Sample data: `data/sample/truth_table_sample.csv`
- Example analysis notebook: `notebooks/Example_Analysis.ipynb`

To build this documentation locally:
```bash
pip install -r requirements-dev.txt
mkdocs serve  # http://127.0.0.1:8000
```

To deploy (via GitHub Actions):
- Push/merge to `main`. The workflow `.github/workflows/gh-pages.yml` builds and publishes to the `gh-pages` branch.
