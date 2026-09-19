# GPU-Accelerated N-Body Simulation (CUDA)

A high-performance gravitational N-body simulator written in CUDA C++, benchmarking four kernel/data-layout strategies on a simplified solar system with a highly eccentric comet.

The project's focus is **performance**: how thread mapping, memory access patterns, and cache locality affect throughput on modern GPUs, and how those choices compare to a naive baseline.

![Kernel comparison](results/enhanced/N16384/benchmark_comparison.png)

---

## Overview

The N-body problem computes gravitational interactions between every pair of `N` bodies. Because each body interacts with all others, the naive complexity is **O(N²)** — roughly 4.3 billion interactions per step at N = 65,536. That arithmetic intensity makes it an ideal GPU benchmark.

We implement and compare **four kernels**:

| # | Kernel | Data layout | Cache strategy |
|---|--------|-------------|----------------|
| 1 | Naive AoS | Array of Structures | None (global memory reads every inner iteration) |
| 2 | Tiled AoS | Array of Structures | Shared-memory tiling |
| 3 | Naive SoA | Structure of Arrays | None (but coalesced global accesses) |
| 4 | Tiled SoA | Structure of Arrays | Shared-memory tiling **+** coalesced layout |

The **Tiled SoA** variant achieves up to **2.22× speedup** over the naive baseline, with peak throughput of ~345 GFLOPS at N = 16,384.

---

## Physics

**Gravitational force** between bodies i and j:

    F_i = Σ_j≠i  G·m_i·m_j / (|r_ij|² + ε²)^(3/2) · r_ij

**Softening** ε = 10⁹ m prevents singularities when two bodies pass very close.

**Integrator**: Velocity Verlet — a second-order symplectic scheme with excellent long-term energy conservation.

    v(t+Δt/2) = v(t) + a(t)·Δt/2
    r(t+Δt)   = r(t) + v(t+Δt/2)·Δt
    a(t+Δt)   = compute_forces(r(t+Δt))
    v(t+Δt)   = v(t+Δt/2) + a(t+Δt)·Δt/2

**Collisions** (enhanced build only): when two bodies come within `r < 10⁹ m`, they merge inelastically:

    m_new = m_i + m_j
    r_new = (m_i·r_i + m_j·r_j) / m_new        # center of mass
    v_new = (m_i·v_i + m_j·v_j) / m_new        # momentum conservation

---

## System Setup

Initial configuration (created on the CPU before the sim loop):

| Body | Count | Notes |
|------|-------|-------|
| Sun | 1 | Fixed at origin, M = 1.989 × 10³⁰ kg |
| Inner + outer planets | 8 | Circular/elliptical orbits, e < 0.05 |
| **Comet** | 1 | **e ≈ 0.91, perihelion 0.3 AU, aphelion 10 AU** |
| Asteroids | N − 10 | Uniform in [0.4, 8.4] AU, random eccentricity |

Initial velocities use the vis-viva equation: `v = sqrt(G·M_sun · (2/r − 1/a))`.

Simulation parameters:
- Δt = 3600 s (1 hour)
- Steps = 8,760 (1 simulated year)
- Output every 24 steps (1 simulated day)

---

## Benchmark Results

Measured on an NVIDIA RTX 4070 Mobile (4,608 CUDA cores, 36 SMs, 256 GB/s).

### Kernel execution time (N = 16,384)

| Variant | Time (ms) | GFLOPS | Speedup |
|---------|-----------|--------|---------|
| Naive AoS | 51.2 | 10.7 | 1.00× (baseline) |
| Naive SoA | 40.1 | 13.7 | 1.28× |
| Tiled AoS | 25.3 | 21.7 | 2.02× |
| **Tiled SoA** | **23.1** | **23.7** | **2.22×** |

### Key observations

- **SoA alone** gives ~28% from memory coalescing.
- **Tiling alone** gives ~2× from shared-memory caching.
- **Combined** they multiply: ~2.22× total.
- The kernel is **memory-bandwidth-bound**, not compute-bound — peak GFLOPS is only ~2% of the GPU's theoretical FP32 peak.
- Optimal N range: 8K–16K bodies. Below that, launch overhead dominates; above, bandwidth saturates.

### Physics validation

- Energy drift < **0.01%** over one simulated year
- Orbital periods remain stable; semi-major axes drift < 0.5%
- Comet orbit matches Kepler's laws (perihelion 0.58 AU, aphelion 12.7 AU, e = 0.912)

---

## Repository Structure

    .
    ├── src/
    │   ├── nbody_baseline.cu     # Naive + Tiled (AoS only), no collisions
    │   └── nbody_enhanced.cu     # 4 kernel variants + collisions + merging
    ├── scripts/
    │   └── visualize.py          # Reads CSVs, produces PNGs and GIFs
    ├── docs/
    │   └── report.pdf            # Full written report
    └── results/
        ├── baseline/             # Outputs of nbody_baseline.cu
        └── enhanced/             # Outputs of nbody_enhanced.cu

---

## Requirements

- CUDA Toolkit ≥ 12.0 (`nvcc`)
- A CUDA-capable GPU (tested on sm_89 / RTX 4070)
- Python 3.9+ with `numpy`, `matplotlib`, `pandas`, `imageio` (for visualization)

On Ubuntu:

    sudo apt install -y nvidia-cuda-toolkit python3-pip
    pip install numpy matplotlib pandas imageio

---

## Build and Run

### Baseline

    nvcc -O3 -use_fast_math -o nbody_baseline src/nbody_baseline.cu
    ./nbody_baseline 16384     # N = 16,384 bodies

### Enhanced

    nvcc -O3 -use_fast_math -o nbody_enhanced src/nbody_enhanced.cu
    ./nbody_enhanced 16384

Both binaries write CSV files into `output/`:
- `positions.csv`, `energy.csv`, `performance.csv`, `comet_track.csv`
- (enhanced only) `collisions.csv`, `active_bodies.csv`

### Visualize

    python3 scripts/visualize.py

Reads from `output/` and writes PNG/GIF figures.

---

## Implementation Notes

### Thread mapping

- One thread per body: `thread i` accumulates forces from all `j`
- Block size = 256
- Grid size = `ceil(N / 256)`

### Shared-memory tiling

Each block cooperatively loads a tile of 256 bodies into shared memory, then every thread reads from that tile. Global memory traffic drops from `O(N²)` to `O(N² / 256)`.

### Why SoA matters

In AoS, a warp accessing `body[j].x` for 32 different `j` straddles 32 different 32-byte structs → 32 memory transactions. In SoA, the same access is 32 consecutive floats → 1 coalesced 128-byte transaction.

---

## Authors

- Seyed Ehsan Mousavi
- Mojtaba Zandavi

Course project, February 2026.

---

## License

MIT — see `LICENSE`.
