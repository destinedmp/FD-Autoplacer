# FD-Autoplacer

**A GPU-accelerated, force-directed PCB component autoplacer for KiCad 6+ boards.**

It reads a `.kicad_pcb` file, optimizes component placement using differentiable gradient descent on PyTorch tensors, and writes the result back to a new `.kicad_pcb` file — ready to open directly in KiCad and hand off to an autorouter.

---

## How It Works

### The Core Idea

Placement is formulated as a **continuous optimization problem**: given $N$ components with positions $(x_i, y_i, \theta_i)$, find the configuration that minimizes a weighted sum of differentiable cost terms. Because every term is smooth, PyTorch's autograd computes exact analytical gradients and the AdamW optimizer descends toward the minimum in seconds.

This is a significant step up from classic force-directed placement: instead of applying simplified spring forces in discrete Euler steps and hoping for convergence, the full gradient of the entire objective is computed and used directly.

---

## Architecture

```
main.py              Entry point, KiCad parser, visualization
optim_engine.py      GPU/CPU differentiable placement engine (PyTorch)
kicad_exporter.py    S-expression writer — exports placed file back to KiCad
tests/               Sample .kicad_pcb boards
```

### `main.py` — Parser & Optimization Driver

Contains two core parts:

1. **KiCad 6+ S-expression parser** (`parse_sexp`, `import_kicad_pcb`)
   - Parses `footprint`, `pad`, `net`, `fp_text`, `gr_line`, `gr_arc`, `gr_rect` nodes
   - Extracts component reference, position, rotation, pad positions, and net names
   - Reconstructs the `Edge.Cuts` board boundary into a polygon using segment chaining
   - Negates the Y axis on read (KiCad uses Y-down; internally Y-up is used throughout)

2. **Visualization & Utilities** (`draw`, `optimize_swap_group`)
   - matplotlib rendering of board polygon, net ratsnest lines, component boxes, and pin locations
   - Hungarian algorithm pin-swap optimization (`scipy.optimize.linear_sum_assignment`) to re-assign equivalent symmetric pins to reduce total wirelength after spatial placement

### `optim_engine.py` — GPU Optimization Engine

#### `PlacementProblem`
Encodes all parsed data into PyTorch tensors on the selected device:
- `half_sizes` `(N, 2)` — component bounding half-extents
- `is_free` `(N,)` — mask for movable vs. fixed components
- `nets` — list of dicts, each containing index arrays and pin local offsets for one net
- `decoup` — heuristically detected (bypass cap → IC) pairs for grouping springs

#### `PlacerModel` (`nn.Module`)
Holds three learnable parameters:

| Parameter | Shape | Description |
|-----------|-------|-------------|
| `X` | `(N,)` | Component X centres |
| `Y` | `(N,)` | Component Y centres |
| `Theta` | `(N,)` | Component rotation (radians) |

**Loss terms:**

| Term | Formula | Purpose |
|------|---------|---------|
| **LSE Wirelength** | `γ·(logsumexp(x/γ) + logsumexp(-x/γ) + …)` | Smooth HPWL approximation — pulls same-net pins together. Power nets weighted 0.05× |
| **Overlap** | `Σ relu(sx−dx)·relu(sy−dy)` over all pairs | Pairwise AABB intersection area using rotation-aware extents |
| **Boundary** | `Σ relu(bmin−corner)² + relu(corner−bmax)²` | Quadratic penalty for exceeding board bounding box |
| **Orientation** | `Σ sin²(2θ)` | Periodic snap with minima at 0°, 90°, 180°, 270° |
| **Decoupling** | `Σ relu(dist − target)²` | Proximity springs between detected bypass caps and their IC |

#### Three-Phase Pipeline

```
Phase 1 — Global Placement (default 1500 iters)
   AdamW + CosineAnnealingLR
   Weight schedule: overlap 10→310, boundary 50→350, LSE gamma 10→1, orient 0→8
   Gradually enforces non-overlap while minimizing wirelength

Phase 2 — Legalization
   Hard-snap every θ to the nearest multiple of 90° (no gradient)

Phase 3 — Refinement (400 iters)
   θ frozen; very high overlap (500×) and boundary (500×) weights
   Resolves any residual collisions while continuing to minimize wirelength
```

After all three phases, `project_position` projects each component centroid into the actual board polygon (the GPU model only enforces the axis-aligned bounding box), and a final Hungarian pin-swap pass is run.

### `kicad_exporter.py` — Round-Trip Writer

- Parses the original `.kicad_pcb` into an S-expression AST
- Walks every `footprint` node, looks up the optimized position from the `Component` dict
- Updates the `(at X Y θ)` field in-place, converting back to KiCad's Y-down coordinate system and degrees
- Serializes the full AST back to a formatted string preserving all non-placement data (tracks, zones, copper fills, text, design rules, etc.)

---

## Installation

```bash
git clone https://github.com/destinedmp/FD-Autoplacer
cd FD-Autoplacer
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

**Dependencies:** `numpy`, `scipy`, `matplotlib`, `torch`

For GPU acceleration, PyTorch will automatically use CUDA if a compatible GPU and drivers are present. CPU fallback is automatic.

---

## Usage

```bash
# Optimize a board (GPU if available, otherwise CPU)
python main.py path/to/board.kicad_pcb

# Force CPU even when CUDA is present
python main.py path/to/board.kicad_pcb --cpu

# Tune the optimizer
python main.py path/to/board.kicad_pcb --iters 2000 --lr 0.03

# Export placed positions back to a KiCad file
python main.py path/to/board.kicad_pcb --export path/to/board_placed.kicad_pcb

# Keep original positions (don't scramble before placing)
python main.py path/to/board.kicad_pcb --no-scramble
```

All runs save a `placement.png` side-by-side comparison of the initial and converged layouts.

### CLI Reference

| Flag | Default | Description |
|------|---------|-------------|
| `pcb` | `tests/simple.kicad_pcb` | Path to input `.kicad_pcb` file |
| `--cpu` | off | Force CPU even when CUDA is available |
| `--iters N` | 1500 | Global placement iterations |
| `--lr F` | 0.05 | AdamW learning rate |
| `--export PATH` | none | Write placed PCB to this file |
| `--no-scramble` | off | Don't randomize positions before placement |

---

## Results

On `tests/leds.kicad_pcb` (99 components, 51 nets, complex cross-shaped board outline):

| Metric | Initial | Converged |
|--------|---------|-----------|
| Wirelength (LSE HPWL) | 4,059 | 2,100 |
| Overlap area | 781 | ~0 |
| Boundary violations | 0 | ~0 |
| Runtime (CPU) | — | ~47 seconds |

---

## Constraints & Design Rules Supported

| Constraint | How it's handled |
|-----------|-----------------|
| Board outline (any polygon) | Parsed from `Edge.Cuts` layer (`gr_line`, `gr_arc`, `gr_rect`). Used for post-GPU polygon projection |
| Fixed components | `fixed=True` — gradients zeroed; positions never updated |
| Allowed orientations | Snapped during legalization and final pass |
| Component keep-in zone | Per-component `allowed_rect` or `allowed_polygon` |
| Pin swapping | Hungarian assignment on equivalent-pin swap groups after placement |


---

## Theory

Classic **force-directed placement** models components as charged particles that repel each other and nets as springs that attract connected pins. The system is simulated until it reaches mechanical equilibrium.

The optimizer in this project replaces the simulation with **direct gradient descent** on a differentiable objective. The key insight is the **Log-Sum-Exp (LSE) wirelength approximation**:

$$\text{HPWL}(e) \approx \gamma \left( \log \sum_{i \in e} e^{x_i/\gamma} + \log \sum_{i \in e} e^{-x_i/\gamma} + \log \sum_{i \in e} e^{y_i/\gamma} + \log \sum_{i \in e} e^{-y_i/\gamma} \right)$$

As $\gamma \to 0$ this converges exactly to $\max(x) - \min(x) + \max(y) - \min(y)$, the true half-perimeter wirelength. Using a finite $\gamma$ keeps the function smooth so gradients flow through it cleanly. The schedule anneals $\gamma$ from 10 down to 1 over the course of optimization.
