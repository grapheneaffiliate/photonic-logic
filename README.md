# Programmable Photonic Logic (v2.2)

The industry's first comprehensive photonic circuit design platform - the "SPICE for photonic logic." Transform from trial-and-error physics experiments to quantitative design with real material parameters, power budgets, and thermal analysis.

## Performance at a Glance

| Platform | Power Required | Energy/Op | Max Cascade | Thermal Safe | CMOS Compatible |
|----------|---------------|-----------|-------------|--------------|-----------------|
| AlGaAs   | 0.67× baseline| 100 fJ    | 8 stages    | <100 mW     | ❌              |
| Silicon  | 2.2× baseline | 330 fJ    | 5 stages    | <10 mW      | ✅              |
| SiN      | 42× baseline  | 500 fJ    | 3-6 stages  | <500 mW     | ✅              |

*Baseline: 1 mW control power with n₂=1e-17 m²/W reference*

## Quick Start (30 Seconds)

```bash
pip install -e .
plogic cascade --platform AlGaAs  # See best-case performance
plogic sweep --platforms Si SiN AlGaAs --csv comparison.csv  # Compare all platforms
plogic cascade --platform SiN --report power  # Detailed power analysis
```

## Why This Matters

**The Gap**: Photonic logic research uses abstract physics models, but real devices need material-specific power budgets, thermal management, and fabrication constraints.

**The Solution**: This platform bridges theory to practice with:
- ✅ **Real material parameters** (Si/SiN/AlGaAs from literature)
- ✅ **Power budget analysis** (energy/op, thermal safety, cascade limits)
- ✅ **Design space exploration** (parameter sweeps, optimization guidance)
- ✅ **Fab-ready validation** (extinction ratios, thermal predictions)

## Installation

### Quick Install (Recommended)
```bash
git clone https://github.com/grapheneaffiliate/photonic-logic.git
cd photonic-logic
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

### Development Install
```bash
pip install -r requirements-dev.txt  # linting/tests/docs
```

### Conda Environment
```bash
conda env create -f environment.yml
conda activate photonic-logic
pip install -e .[dev]
```

## Critical Design Parameters

Understanding these parameters is essential for photonic circuit design:

- **n₂ (Kerr coefficient)**: Determines power requirements via P ∝ 1/n₂
  - AlGaAs: 1.5e-17 m²/W (strong, low power)
  - Silicon: 4.5e-18 m²/W (moderate, CMOS compatible)
  - SiN: 2.4e-19 m²/W (weak, ultra-stable)

- **Power scaling**: Relative power needed vs AlGaAs baseline
- **Cascade depth**: Maximum stages before signal regeneration needed
- **ER margin**: Extinction ratio safety margin for fabrication tolerances
- **Thermal flag**: ok/caution/danger based on thermal vs Kerr effects

## Real-World Design Examples

### Low-Power Dense Logic (AlGaAs)
```bash
plogic cascade --platform AlGaAs --P-high-mW 0.5 --report power
```
- 0.67 mW control power
- 100 fJ/operation
- Suitable for dense integration
- **Caution**: Thermal management critical

### CMOS-Compatible Router (Silicon)
```bash
plogic cascade --platform Si --pulse-ns 0.1 --include-2pa --report power
```
- Sub-100ps switching
- CMOS foundry compatible
- **Watch**: TPA thermal limits above 10 mW

### Ultra-Stable High-Q Logic (SiN)
```bash
plogic cascade --platform SiN --coupling-eta 0.9 --link-length-um 20 --report power
```
- Excellent thermal stability
- 6+ cascade stages possible
- **Trade-off**: Higher power requirements

### Design Space Exploration
```bash
# Platform comparison
plogic sweep --platforms Si SiN AlGaAs --P-high-mW 0.5 1.0 --csv platform_comparison.csv

# Cascade depth optimization
plogic sweep --platforms SiN --coupling-eta 0.8 0.85 0.9 --link-length-um 20 50 100 --csv depth_optimization.csv

# Energy scaling analysis
plogic sweep --platforms Si --P-high-mW 0.3 0.5 0.8 --pulse-ns 0.2 0.5 1.0 --csv energy_scaling.csv
```

## Enhanced CLI Features (v2.2+)

### Material Platform Integration
```bash
# Platform-specific analysis
plogic cascade --platform SiN --report power --auto-timing

# Parameter debugging
plogic cascade --platform Si --show-resolved --include-2pa

# Override platform defaults
plogic cascade --platform Si --n2 3e-18 --q-factor 5e5
```

### Power Budget Analysis
```bash
# Comprehensive power reporting
plogic cascade --platform SiN --report power --P-high-mW 0.5

# Energy optimization
plogic cascade --platform AlGaAs --pulse-ns 0.3 --report power

# Thermal safety analysis
plogic cascade --platform Si --include-2pa --P-high-mW 1.0 --report power
```

### Design Space Exploration
```bash
# Multi-platform sweep
plogic sweep --platforms Si SiN AlGaAs --P-high-mW 0.5 1.0 --csv results.csv

# Optimization campaigns
plogic sweep --platforms SiN --beta 80 100 --coupling-eta 0.8 0.9 --csv optimization.csv
```

## Photonics vs Electronics Comparison

| Metric | Photonic Logic | 7nm CMOS | Advantage |
|--------|---------------|----------|-----------|
| Energy/Op | 100-500 fJ | 50-200 fJ | Comparable |
| Speed | 1-10 GHz | 1-5 GHz | Photonics edge |
| Density | 100-1000 gates/mm² | 10M+ gates/mm² | Electronics wins |
| Static Power | 0 W | μW-mW | **Photonics wins** |
| Wavelength Mux | Yes | No | **Photonics unique** |
| Thermal | Critical | Managed | Electronics mature |

**Photonic Advantage**: Zero static power + wavelength multiplexing enable new architectures impossible in electronics.

## Known Limitations

**Current Constraints** (honest engineering assessment):
- **Cascade depth**: Limited by power decay and thermal effects (3-8 stages typical)
- **Two-photon absorption**: Limits Silicon to <10 mW operation
- **SiN power requirements**: 42× higher than AlGaAs due to weak Kerr effect
- **Thermal management**: Critical above 100 mW/mm² for all platforms
- **Fabrication tolerance**: ±50 pm wavelength precision required

**Roadmap Items**:
- [ ] Enhanced thermal dynamics modeling
- [ ] Fabrication tolerance Monte Carlo analysis
- [ ] Integration with gdsfactory for layout generation
- [ ] Quantum stretch goals (Rydberg EIT)

## Troubleshooting Guide

### "Thermal flag: danger"
**Problem**: Thermal effects dominate Kerr effects
**Solutions**:
```bash
# Reduce drive power
plogic cascade --platform Si --P-high-mW 0.5

# Use shorter pulses
plogic cascade --platform AlGaAs --pulse-ns 0.3

# Switch to thermally stable platform
plogic cascade --platform SiN --report power
```

### Poor extinction ratio
**Problem**: `meets_extinction: false` or low contrast margin
**Solutions**:
```bash
# Adjust threshold for better margins
plogic cascade --platform SiN --threshold-norm 0.55

# Use steeper sigmoid
plogic cascade --platform Si --beta 100

# Check precise margins
plogic cascade --platform AlGaAs --report power  # See contrast_breakdown
```

### Limited cascade depth
**Problem**: `max_depth_meeting_thresh` too low
**Solutions**:
```bash
# Improve coupling efficiency
plogic cascade --platform SiN --coupling-eta 0.9

# Reduce link lengths
plogic cascade --platform Si --link-length-um 20

# Increase drive power (if thermal allows)
plogic cascade --platform SiN --P-high-mW 1.0
```

## Advanced Features

### Power Budget Analysis
The `--report power` flag provides comprehensive analysis:
- **Energy per operation**: fJ calculations with photon counting
- **Thermal safety**: ok/caution/danger flags based on physics
- **Cascade limits**: Maximum stages before signal degrades
- **Extinction validation**: Meets target ER requirements
- **Contrast breakdown**: Engineering margins for fab validation

### Design Space Exploration
Parameter sweeps enable rapid optimization:
- **Platform comparison**: Quantitative Si vs SiN vs AlGaAs trade-offs
- **Power optimization**: Energy scaling with drive power and timing
- **Cascade analysis**: Depth limits vs coupling and link parameters

### Quality-of-Life Features
- `--show-resolved`: Debug parameter resolution
- `--embed-report`: Single JSON artifact
- `--quiet`: CI-friendly operation
- `--csv`: Spreadsheet-ready export

## What's New in v2.2 (Material Platform Integration)

**Revolutionary Enhancement**: Transform from physics simulator to production design platform.

### Material Platform Database
- **Si/SiN/AlGaAs**: Literature-backed parameters with validation
- **Platform selection**: `--platform Si|SiN|AlGaAs`
- **Parameter override**: Flags > platform > defaults hierarchy

### Power Budget Analysis
- **Energy calculations**: fJ per operation with photon counting
- **Thermal analysis**: Physics-based safety flags
- **Cascade depth**: Real power decay and fanout limits
- **Measured statistics**: Uses actual ON/OFF levels, not assumptions

### Design Space Exploration
- **Parameter sweeps**: Multi-platform grid exploration
- **CSV export**: Consolidated data for analysis
- **Parallel processing**: Scalable design space mapping

## Contributing

We welcome contributions! Priority areas:
- **Material platforms**: New materials with literature citations
- **Validation**: Power measurements from real devices
- **Thermal models**: Improved heat dissipation analysis
- **Bug fixes**: With comprehensive test coverage

## Citation

If you use this framework for research or commercial development, please cite:

```bibtex
@software{photonic_logic_2024,
  title = {Photonic Logic: A Practical Framework for All-Optical Computing},
  author = {Open Photonics Lab},
  year = {2024},
  url = {https://github.com/grapheneaffiliate/photonic-logic},
  note = {Material platforms, power analysis, design space exploration}
}
```

## License

MIT License. See `LICENSE`.

---

**Ready to revolutionize photonic circuit design?** This platform provides everything needed to go from concept to fab-ready validation. Start with the quick examples above, then explore the design space with parameter sweeps and power analysis.

**The "SPICE for photonic logic" is here.** 🚀
