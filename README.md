# Programmable Photonic Logic (v2.3)

The industry's first comprehensive photonic circuit design platform - the "SPICE for photonic logic." Transform from trial-and-error physics experiments to quantitative design with real material parameters, power budgets, thermal analysis, and now with **parallel fanout capabilities** and **hybrid material platforms**.

## 🆕 What's New in v2.3

### Fanout Parallelism
- **Parallel processing**: Split signals to multiple gates with configurable loss
- **Depth reduction**: Effective cascade depth ~depth/√fanout
- **Energy scaling**: Total energy = E_op × fanout for parallel operations
- **Configurable split loss**: Default 0.5 dB per split, adjustable via CLI

### Hybrid Material Platforms
- **AlGaAs/SiN integration**: Logic in AlGaAs (1.0 dB/cm), routing in SiN (0.1 dB/cm)
- **Smart material switching**: Optimize power vs loss trade-offs
- **Mode converter modeling**: 0.2 dB loss per transition, 0.95 coupling efficiency
- **Configurable routing fraction**: Control material usage balance

### Realistic vs Idealized Modes
- **Realistic mode**: Use measured extinction ratios for fabrication
- **Idealized mode**: Theoretical limits for research exploration
- **CLI warnings**: Automatic alerts when using idealized assumptions
- **Documentation**: Comprehensive guide in `docs/LIMITATIONS_AND_ROADMAP.md`

## Performance at a Glance

| Platform | Power Required | Energy/Op | Max Cascade | Thermal Safe | CMOS Compatible |
|----------|---------------|-----------|-------------|--------------|-----------------|
| AlGaAs   | 0.06 mW*      | 84 fJ**   | 33 stages***| <1 mW       | ❌              |
| Silicon  | 2.2× baseline | 330 fJ    | 5 stages    | <10 mW      | ✅              |
| SiN      | 42× baseline  | 500 fJ    | 3-6 stages  | <500 mW     | ✅              |
| **Hybrid**| Variable     | Optimized | 10+ stages  | Balanced    | ✅              |

*\*Optimized with η=0.98 coupling, 60µm links, 1.4ns pulses*  
*\*\*Ultra-low energy with optimized parameters*  
*\*\*\*11× improvement with proper thermal calculations and optimization*

**Why material selection matters:**
- One AlGaAs gate: 0.67 mW (but thermally limited)
- Same gate in SiN: 42 mW (**62× more power, but thermally stable!**)
- Hybrid approach: Balance power and stability optimally

## Quick Start (30 Seconds)

```bash
pip install -e .

# Basic optimized cascade
plogic demo --gate XOR --platform AlGaAs --P-high-mW 0.06 --pulse-ns 1.4 --coupling-eta 0.98 --link-length-um 60

# NEW: Fanout parallelism (v2.3)
plogic cascade --platform AlGaAs --fanout 4 --split-loss-db 0.5 --report power

# NEW: Hybrid platform (v2.3)
plogic cascade --hybrid --routing-fraction 0.7 --report power

# Platform comparison
plogic sweep --platforms Si --platforms SiN --platforms AlGaAs --csv comparison.csv
```

### 🎯 The "Holy Grail" Commands

#### Classic Single-Path Logic
```bash
plogic demo --gate XOR --platform SiN --threshold soft --output truth-table
```

#### NEW: Parallel Fanout Logic (v2.3)
```bash
plogic cascade --platform AlGaAs --fanout 4 --split-loss-db 0.3 --report power
```
**Demonstrates parallel processing with 4× fanout, reducing effective depth by ~2×**

#### NEW: Hybrid Material Routing (v2.3)
```bash
plogic cascade --hybrid --routing-fraction 0.7 --report power
```
**Shows optimal material switching: 30% logic in AlGaAs (default), 70% routing in low-loss SiN (default)**

## Why This Matters

**The Gap**: Photonic logic research uses abstract physics models, but real devices need material-specific power budgets, thermal management, fabrication constraints, and now **scalable architectures**.

**The Solution**: This platform bridges theory to practice with:
- ✅ **Real material parameters** (Si/SiN/AlGaAs from literature)
- ✅ **Power budget analysis** (energy/op, thermal safety, cascade limits)
- ✅ **Design space exploration** (parameter sweeps, optimization guidance)
- ✅ **Fab-ready validation** (extinction ratios, thermal predictions)
- 🆕 **Parallel architectures** (fanout splitting, depth reduction)
- 🆕 **Hybrid platforms** (material switching for optimization)

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

### Material Properties
- **n₂ (Kerr coefficient)**: Determines power requirements via P ∝ 1/n₂
  - AlGaAs: 1.5e-17 m²/W (strong, low power)
  - Silicon: 4.5e-18 m²/W (moderate, CMOS compatible)
  - SiN: 2.4e-19 m²/W (weak, ultra-stable)

### NEW: Fanout Parameters (v2.3)
- **Fanout degree**: Number of parallel paths (1-8 typical)
- **Split loss**: Power loss per split (0.3-1.0 dB typical)
- **Split efficiency**: η_split = 10^(-split_loss_db/10)
- **Effective depth**: depth_eff = max(1, int(n_stages / √fanout))

### NEW: Hybrid Platform Parameters (v2.3)
- **Logic material**: High nonlinearity for switching (AlGaAs default)
- **Routing material**: Low loss for interconnects (SiN default)
- **Routing fraction**: Percentage of path in routing material (0.0-1.0)
- **Mode converter loss**: Loss at material interfaces (0.2 dB typical)
- **Coupling efficiency**: Power transfer between materials (0.95 typical)

### Performance Metrics
- **Power scaling**: Relative power needed vs AlGaAs baseline
- **Cascade depth**: Maximum stages before signal regeneration needed
- **ER margin**: Extinction ratio safety margin for fabrication tolerances
- **Thermal flag**: ok/caution/danger based on thermal vs Kerr effects

## Real-World Design Examples

### Low-Power Dense Logic (AlGaAs) - Optimized
```bash
plogic cascade --platform AlGaAs --P-high-mW 0.06 --pulse-ns 1.4 --coupling-eta 0.98 --link-length-um 60 --report power
```
- 0.06 mW control power (ultra-low)
- 84 fJ/operation
- **33-stage cascade depth** (revolutionary improvement)
- Thermal ratio: 0.45 (safe operation)

### NEW: Parallel Processing Network (v2.3)
```bash
plogic cascade --platform AlGaAs --fanout 4 --split-loss-db 0.5 --stages 8 --report power
```
- 4× parallel fanout
- Effective depth: 4 stages (reduced from 8)
- Total energy: 336 fJ (84 fJ × 4)
- Enables complex parallel architectures

### NEW: Hybrid Long-Distance Router (v2.3)
```bash
plogic cascade --hybrid --routing-fraction 0.8 --report power
```
- 20% AlGaAs for logic operations (default)
- 80% SiN for low-loss routing (default)
- 10+ cascade stages possible
- Balanced power and loss optimization

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
plogic sweep --platforms Si --platforms SiN --platforms AlGaAs --P-high-mW 0.5 --P-high-mW 1.0 --csv platform_comparison.csv

# NEW: Fanout optimization (v2.3)
plogic sweep --platforms AlGaAs --fanout 1 --fanout 2 --fanout 4 --fanout 8 --split-loss-db 0.3 --split-loss-db 0.5 --split-loss-db 1.0 --csv fanout_analysis.csv

# NEW: Hybrid routing optimization (v2.3)
plogic sweep --platforms AlGaAs --routing-fraction 0.3 --routing-fraction 0.5 --routing-fraction 0.7 --routing-fraction 0.9 --csv hybrid_optimization.csv

# Energy scaling analysis
plogic sweep --platforms Si --P-high-mW 0.3 --P-high-mW 0.5 --P-high-mW 0.8 --pulse-ns 0.2 --pulse-ns 0.5 --pulse-ns 1.0 --csv energy_scaling.csv
```

## Enhanced CLI Features (v2.3+)

### NEW: Fanout Control (v2.3)
```bash
# Basic fanout
plogic cascade --platform AlGaAs --fanout 4 --report power

# Custom split loss
plogic cascade --platform SiN --fanout 2 --split-loss-db 0.3 --report power

# Fanout with optimization
plogic cascade --platform Si --fanout 8 --stages 16 --report power
```

### NEW: Hybrid Platform Control (v2.3)
```bash
# Default hybrid (AlGaAs/SiN)
plogic cascade --hybrid --report power

# Routing fraction control (adjusts balance between AlGaAs logic and SiN routing)
plogic cascade --hybrid --routing-fraction 0.6 --report power

# With fanout for parallel processing
plogic cascade --hybrid --routing-fraction 0.7 --fanout 4 --report power
```

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

# Energy optimization with fanout
plogic cascade --platform AlGaAs --fanout 4 --pulse-ns 0.3 --report power

# Thermal safety with hybrid
plogic cascade --hybrid --P-high-mW 1.0 --report power
```

## Photonics vs Electronics Comparison

| Metric | Photonic Logic | 7nm CMOS | Advantage |
|--------|---------------|----------|-----------|
| Energy/Op | 100-500 fJ | 50-200 fJ | Comparable |
| Speed | 1-10 GHz | 1-5 GHz | Photonics edge |
| Density | 100-1000 gates/mm² | 10M+ gates/mm² | Electronics wins |
| Static Power | 0 W | μW-mW | **Photonics wins** |
| Wavelength Mux | Yes | No | **Photonics unique** |
| **Fanout** | 1-8 typical | Unlimited | Electronics mature |
| **Routing Loss** | 0.1-1 dB/cm | ~0 | Electronics wins |
| Thermal | Critical | Managed | Electronics mature |

**Photonic Advantage**: Zero static power + wavelength multiplexing + now parallel fanout architectures enable new computing paradigms.

## Known Limitations & Roadmap

### Current Constraints (honest engineering assessment):
- **Idealized mode**: May overestimate performance - use realistic mode for fabrication
- **Fanout limits**: Practical fanout limited to ~8 due to splitting losses
- **Hybrid transitions**: Mode converter losses (0.2 dB) impact short links
- **AND gate logic**: Outputs [0,0,1,1] instead of [0,0,0,1] at cascade depths 3-4
- **Two-photon absorption**: Limits Silicon to <10 mW operation
- **Thermal management**: Critical above 100 mW/mm² for all platforms

### Roadmap & Next Steps:
- ✅ **Fanout >1 parallelism** (COMPLETED in v2.3)
- ✅ **Hybrid material platforms** (COMPLETED in v2.3)
- ✅ **Realistic/idealized modes** (COMPLETED in v2.3)
- [ ] Enhanced thermal dynamics modeling
- [ ] Fabrication tolerance Monte Carlo analysis
- [ ] Integration with gdsfactory for layout generation
- [ ] Wavelength division multiplexing (WDM) support
- [ ] Quantum stretch goals (Rydberg EIT)

**📖 See `docs/LIMITATIONS_AND_ROADMAP.md` for detailed technical discussion and implementation roadmap.**

### New Documentation & Examples:
- **Limitations & Roadmap**: [`docs/LIMITATIONS_AND_ROADMAP.md`](docs/LIMITATIONS_AND_ROADMAP.md) - Comprehensive guide to current limitations and future improvements
- **Fanout & Hybrid Demo**: [`examples/demo_fanout_hybrid.py`](examples/demo_fanout_hybrid.py) - Proof-of-concept implementation
- **Hybrid Platform Module**: [`src/plogic/materials/hybrid.py`](src/plogic/materials/hybrid.py) - Core hybrid platform support
- **Test Suite**: [`tests/test_fanout_hybrid.py`](tests/test_fanout_hybrid.py) - Validation tests for new features

## Troubleshooting Guide

### "Thermal flag: danger"
**Problem**: Thermal effects dominate Kerr effects
**Solutions**:
```bash
# Reduce drive power
plogic cascade --platform Si --P-high-mW 0.5

# Use hybrid platform for better thermal management
plogic cascade --hybrid --routing-fraction 0.7

# Switch to thermally stable platform
plogic cascade --platform SiN --report power
```

### Poor extinction ratio with fanout
**Problem**: Signal degradation with high fanout
**Solutions**:
```bash
# Reduce fanout degree
plogic cascade --platform AlGaAs --fanout 2 --report power

# Optimize split loss
plogic cascade --platform SiN --fanout 4 --split-loss-db 0.3

# Use hybrid platform for better signal preservation
plogic cascade --hybrid --fanout 4 --report power
```

### Limited cascade depth
**Problem**: `max_depth_meeting_thresh` too low
**Solutions**:
```bash
# Use fanout to reduce effective depth
plogic cascade --platform AlGaAs --fanout 4 --stages 8

# Switch to hybrid platform
plogic cascade --hybrid --routing-fraction 0.6

# Improve coupling efficiency
plogic cascade --platform SiN --coupling-eta 0.95
```

## Advanced Features

### Power Budget Analysis
The `--report power` flag provides comprehensive analysis:
- **Energy per operation**: fJ calculations with photon counting
- **Fanout energy scaling**: Total energy with parallel operations
- **Hybrid loss analysis**: Material transition impacts
- **Thermal safety**: ok/caution/danger flags based on physics
- **Cascade limits**: Maximum stages before signal degrades
- **Extinction validation**: Meets target ER requirements

### Design Space Exploration
Parameter sweeps enable rapid optimization:
- **Platform comparison**: Quantitative Si vs SiN vs AlGaAs vs Hybrid trade-offs
- **Fanout optimization**: Parallelism vs energy trade-offs
- **Hybrid tuning**: Logic/routing material balance
- **Power optimization**: Energy scaling with drive power and timing
- **Cascade analysis**: Depth limits vs coupling and link parameters

### Quality-of-Life Features
- `--show-resolved`: Debug parameter resolution
- `--embed-report`: Single JSON artifact
- `--quiet`: CI-friendly operation
- `--csv`: Spreadsheet-ready export
- `--fanout`: Easy parallel architecture exploration
- `--hybrid`: Quick hybrid platform testing

## What's New in v2.3 (Parallel & Hybrid Architectures)

**Game-Changing Enhancement**: From single-path to parallel architectures with hybrid material optimization.

### Fanout Parallelism
- **Configurable fanout**: 1-8 parallel paths with `--fanout`
- **Split loss modeling**: Realistic power division with `--split-loss-db`
- **Depth reduction**: Automatic calculation of effective cascade depth
- **Energy scaling**: Proper accounting for parallel operation costs

### Hybrid Material Platforms
- **AlGaAs/SiN default**: Optimized for logic vs routing trade-offs
- **Custom combinations**: Any material pairing supported
- **Mode converter modeling**: Realistic transition losses
- **Routing fraction control**: Fine-tune material usage

### Enhanced Analysis
- **Realistic vs idealized**: Clear distinction for fab vs research
- **Comprehensive testing**: 26 new tests for fanout/hybrid features
- **Example demonstrations**: Ready-to-run scripts in `examples/`
- **Updated documentation**: Complete guide in `docs/LIMITATIONS_AND_ROADMAP.md`

## Contributing

We welcome contributions! Priority areas:
- **WDM support**: Wavelength division multiplexing
- **Layout generation**: Integration with gdsfactory
- **Material platforms**: New materials with literature citations
- **Validation**: Power measurements from real devices
- **Thermal models**: Improved heat dissipation analysis

## Citation

If you use this framework for research or commercial development, please cite:

```bibtex
@software{photonic_logic_2024,
  title = {Photonic Logic: A Practical Framework for All-Optical Computing},
  author = {Open Photonics Lab},
  year = {2024},
  version = {2.3},
  url = {https://github.com/grapheneaffiliate/photonic-logic},
  note = {Parallel fanout, hybrid platforms, realistic/idealized modes}
}
```

## License

MIT License. See `LICENSE`.

---

**Ready to revolutionize photonic circuit design?** Version 2.3 brings parallel architectures and hybrid material platforms to enable scalable photonic computing. Start with the quick examples above, explore fanout parallelism, and optimize with hybrid material routing.

**The future of photonic computing is parallel and hybrid.** 🚀
