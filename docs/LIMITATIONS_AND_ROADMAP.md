# Limitations and Roadmap for Photonic-Logic Toolkit

## Current Limitations

### 1. Extinction Ratio Modes

The toolkit supports two modes for extinction ratio (ER) calculations:

- **Realistic Mode** (default): Uses measured statistics from simulations, reflecting actual physics including leakage (e.g., worst_off_norm=0.01 → 20 dB contrast). This mode is ideal for engineering assessments and fabrication planning.
- **Idealized Mode**: Assumes a theoretical floor (e.g., worst_off_norm=1e-12 → 120 dB contrast). Useful for presentations and theoretical demonstrations but may significantly overestimate real-world performance.

⚠️ **WARNING**: Idealized mode assumes perfect conditions and does not reflect fabrication realities. Always use realistic mode for production designs and engineering assessments.

#### Side-by-Side Comparison (Example: XOR Gate on AlGaAs Platform)

| Metric                | Realistic Mode | Idealized Mode | Notes |
|-----------------------|----------------|----------------|-------|
| Floor Contrast (dB)   | 20.0          | 120.0         | Realistic shows actual leakage |
| Target Margin (dB)    | -1.0          | 99.0          | Realistic may not meet target |
| Meets Extinction      | False         | True          | Critical for fab decisions |
| Worst Off Norm        | 0.01          | 1e-12         | 11 orders of magnitude difference |
| Use Case              | Engineering/Fab| Demo/Theory   | Choose based on purpose |
| Reliability           | High          | Low           | Realistic matches fab results |

### 2. Fanout Limitation

**Current State**: The system operates with fanout=1 (single output per gate), limiting parallelism and requiring sequential processing.

**Impact**:
- Sequential processing increases cascade depth
- No parallel computation capability
- Limited throughput for complex logic operations
- Energy inefficiency for operations requiring multiple copies

### 3. Missing SiN Integration

**Current State**: Limited to single-material systems (typically AlGaAs or similar high-index materials).

**Missing Benefits**:
- No hybrid low-loss routing (SiN: ~0.1 dB/cm vs AlGaAs: ~1.0 dB/cm)
- Cannot leverage material-specific advantages
- Higher overall system losses
- Limited design flexibility

## Visual Architecture Diagrams

### Fanout Impact on Cascade Depth

```
Current (Fanout=1):
┌──────┐    ┌──────┐    ┌──────┐
│Gate 1│───►│Gate 2│───►│Gate 3│  Depth = 33 stages
└──────┘    └──────┘    └──────┘

Proposed (Fanout=2):
┌──────┐    ┌──────┐
│Gate 1│───►│Gate 2│  Depth ≈ 16-20 stages
└───┬──┘    └──────┘  (with 0.5 dB split loss)
    │       ┌──────┐
    └──────►│Gate 3│  Parallel processing enabled
            └──────┘

Fanout=3+ Benefits:
- Depth reduction: ~33/√fanout
- Energy cost: E_op × fanout (total)
- Throughput: fanout × single path
```

### Hybrid Material Routing Architecture

```
Current Single-Material System:
[AlGaAs Logic] ──1.0 dB/cm──► [AlGaAs Logic] ──1.0 dB/cm──► [AlGaAs Logic]
                High loss                      High loss

Proposed Hybrid System:
[AlGaAs Logic] ──► [SiN Router] ──0.1 dB/cm──► [AlGaAs Logic]
    (XPM)         (Low-loss routing)              (XPM)
    
Benefits:
- 10× lower propagation loss in routing sections
- Maintain high XPM efficiency in logic sections
- Optimal material usage for each function
```

## Roadmap

### Phase 1: Short-Term (Next Release - v2.0)
- [x] Document limitations and provide clear guidance
- [ ] Implement fanout>1 capability with configurable splitting loss
- [ ] Add fanout-adjusted energy metrics (E_op × fanout)
- [ ] Create HybridPlatform class for material switching
- [ ] Add CLI warnings for idealized mode usage
- [ ] Comprehensive test suite for new features

### Phase 2: Medium-Term (v2.1-2.5)
- [ ] Routing optimization algorithms for minimal loss paths
- [ ] Multi-material parameter sweeps
- [ ] Advanced splitting topologies (tree, mesh)
- [ ] Thermal crosstalk modeling for dense integration
- [ ] Power budget optimization tools

### Phase 3: Long-Term (v3.0+)
- [ ] Complete noise modeling (shot, thermal, 1/f)
- [ ] Process variation analysis
- [ ] GDS-II export for fabrication
- [ ] Integration with photonic EDA tools
- [ ] Machine learning optimization for gate placement

## Implementation Guidelines

### For Engineers
1. Always use `--realistic-extinction` for production designs
2. Include 3-5 dB margin for fabrication tolerances
3. Consider fanout requirements early in design phase
4. Plan for hybrid integration from the start

### For Researchers
1. Document which mode was used in publications
2. Provide both realistic and idealized results when relevant
3. Consider fanout impact on system-level metrics
4. Explore hybrid architectures for optimal performance

### For Software Developers
1. Default to realistic mode in all new features
2. Maintain backward compatibility with fanout=1
3. Ensure clear separation between material systems
4. Add comprehensive tests for all configurations

## Performance Targets

### Current Performance (v1.x)
- Extinction Ratio: 20 dB (realistic)
- Cascade Depth: 33 stages
- Energy/Operation: ~1 pJ
- Propagation Loss: 1.0 dB/cm

### Target Performance (v2.0)
- Extinction Ratio: 21+ dB (realistic, optimized)
- Cascade Depth: 16-20 stages (fanout=2)
- Energy/Operation: ~0.5 pJ (with optimization)
- Propagation Loss: 0.3 dB/cm (hybrid average)

### Ultimate Goals (v3.0+)
- Extinction Ratio: 25+ dB (realistic, advanced materials)
- Cascade Depth: <10 stages (fanout=4+)
- Energy/Operation: <100 fJ
- Propagation Loss: <0.2 dB/cm (optimized hybrid)

## References and Resources

- Extinction Ratio Measurements: [IEEE Photonics Standards]
- SiN Platform Specifications: [AIM Photonics PDK]
- Hybrid Integration Techniques: [Nature Photonics Reviews]
- Fanout Architectures: [Optical Computing Literature]

---

*Last Updated: 2025-08-28*
*Version: 1.0*
*Contact: photonic-logic-dev@example.com*
