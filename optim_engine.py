"""
GPU-accelerated PCB placement optimizer using PyTorch.

Formulates component placement as a differentiable optimization problem:

  minimize   w_wl · L_wirelength  +  w_ov · L_overlap  +  w_bd · L_boundary
           + w_or · L_orientation  +  w_dc · L_decoupling

where every term is smooth and analytically differentiable so that PyTorch
autograd can compute exact gradients and we can use AdamW + cosine-annealing
to converge in seconds.

Loss terms
----------
- **LSE wirelength**: Log-Sum-Exp smooth approximation of the Half-Perimeter
  Wirelength (HPWL) for every net.  As the smoothing parameter γ → 0 the
  approximation becomes exact.
- **Overlap penalty**: Pairwise AABB intersection area, computed with
  rotation-aware bounding boxes (|hx·cosθ| + |hy·sinθ|).
- **Boundary penalty**: Quadratic penalty for component AABBs exceeding the
  board bounding box.  Fine polygon projection is done post-optimization.
- **Orientation snap**: sin²(2θ) has minima at 0°, 90°, 180°, 270°.
- **Decoupling cap grouping**: Strong proximity springs between small bypass
  capacitors and their nearest IC sharing a power net.

Pipeline
--------
Phase 1  Global Placement – full gradient descent with weight annealing
Phase 2  Legalization     – hard-snap θ to nearest 90°
Phase 3  Refinement       – short optimization with frozen θ, high overlap weight
"""

from __future__ import annotations

import time
import numpy as np

try:
    import torch
    import torch.nn as nn

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# ──────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────

POWER_NETS = frozenset(
    {
        "GND",
        "VCC",
        "+5V",
        "+3V3",
        "+3.3V",
        "+12V",
        "VBUS",
        "VDD",
        "VSS",
        "AVCC",
        "AGND",
        "DVCC",
        "DGND",
    }
)


def _is_power(name: str) -> bool:
    return name.upper() in POWER_NETS or name.upper().startswith("+")


def _select_device(pref: str) -> "torch.device":
    if not HAS_TORCH:
        raise ImportError(
            "PyTorch is required for GPU placement.  Install with:  pip install torch"
        )
    if pref == "cuda" and torch.cuda.is_available():
        dev = torch.device("cuda")
        print(f"  [GPU] Using: {torch.cuda.get_device_name(0)}")
    else:
        dev = torch.device("cpu")
        if pref == "cuda":
            print("  [!] CUDA not available -- falling back to CPU")
        else:
            print("  [CPU] Using CPU")
    return dev


# ──────────────────────────────────────────────────────────────────────
# Problem encoding
# ──────────────────────────────────────────────────────────────────────


class PlacementProblem:
    """Encode parsed PCB components and nets into dense PyTorch tensors."""

    def __init__(
        self,
        components: dict,
        board: np.ndarray,
        device_pref: str = "cuda",
    ):
        self.device = _select_device(device_pref)
        self.all_ids = list(components.keys())
        self.n = len(self.all_ids)
        self._g = {cid: i for i, cid in enumerate(self.all_ids)}

        # Component geometry
        self.half_sizes = torch.tensor(
            np.array([components[c].half_size for c in self.all_ids]),
            dtype=torch.float32,
            device=self.device,
        )
        self.is_free = torch.tensor(
            [not components[c].fixed for c in self.all_ids],
            dtype=torch.bool,
            device=self.device,
        )

        # Board polygon & axis-aligned bounding box
        self.board_poly = torch.tensor(board, dtype=torch.float32, device=self.device)
        self.board_min = self.board_poly.min(0).values
        self.board_max = self.board_poly.max(0).values

        # ── Nets ──
        nd: dict[str, list] = {}
        for cid, comp in components.items():
            gi = self._g[cid]
            for pin in comp.pins:
                nd.setdefault(pin.net, []).append(
                    (gi, float(pin.local_pos[0]), float(pin.local_pos[1]))
                )

        self.nets: list[dict] = []
        for name, members in nd.items():
            if len(members) < 2:
                continue
            self.nets.append(
                {
                    "name": name,
                    "idx": torch.tensor(
                        [m[0] for m in members], dtype=torch.long, device=self.device
                    ),
                    "lx": torch.tensor(
                        [m[1] for m in members],
                        dtype=torch.float32,
                        device=self.device,
                    ),
                    "ly": torch.tensor(
                        [m[2] for m in members],
                        dtype=torch.float32,
                        device=self.device,
                    ),
                    "power": _is_power(name),
                }
            )

        # ── Decoupling-cap → IC springs ──
        self.decoup: list[tuple[int, int]] = []
        self._find_decoup(components)

        # Pre-computed upper-triangle mask for pairwise overlap
        self._triu = torch.triu(
            torch.ones(self.n, self.n, dtype=torch.bool, device=self.device),
            diagonal=1,
        )

    # ── heuristic: identify bypass caps ──

    def _find_decoup(self, components: dict) -> None:
        ics: list[tuple[int, object]] = []
        caps: list[tuple[int, object]] = []
        for cid, comp in components.items():
            gi = self._g[cid]
            area = float(comp.half_size[0] * comp.half_size[1] * 4)
            pnets = {p.net.upper() for p in comp.pins}
            has_pwr = bool(pnets & POWER_NETS)
            if area > 8 and has_pwr and not comp.fixed:
                ics.append((gi, comp))
            elif (
                cid.upper().startswith("C")
                and area < 4
                and has_pwr
                and not comp.fixed
            ):
                caps.append((gi, comp))

        for ci, cc in caps:
            best, bd = None, float("inf")
            for ii, ic in ics:
                d = float(np.sum((cc.pos - ic.pos) ** 2))
                if d < bd:
                    bd, best = d, ii
            if best is not None:
                self.decoup.append((ci, best))


# ──────────────────────────────────────────────────────────────────────
# Differentiable placement model
# ──────────────────────────────────────────────────────────────────────


class PlacerModel(nn.Module):
    """Learnable (X, Y, Θ) with differentiable PCB cost functions."""

    def __init__(self, prob: PlacementProblem, components: dict):
        super().__init__()
        self.p = prob

        self.X = nn.Parameter(
            torch.tensor(
                [components[c].pos[0] for c in prob.all_ids],
                dtype=torch.float32,
                device=prob.device,
            )
        )
        self.Y = nn.Parameter(
            torch.tensor(
                [components[c].pos[1] for c in prob.all_ids],
                dtype=torch.float32,
                device=prob.device,
            )
        )
        self.Theta = nn.Parameter(
            torch.tensor(
                [components[c].theta for c in prob.all_ids],
                dtype=torch.float32,
                device=prob.device,
            )
        )

    # ── helpers ──

    def _pin_world(self, net: dict):
        """Compute world-space pin positions for a given net."""
        i = net["idx"]
        ct = torch.cos(self.Theta[i])
        st = torch.sin(self.Theta[i])
        wx = self.X[i] + ct * net["lx"] - st * net["ly"]
        wy = self.Y[i] + st * net["lx"] + ct * net["ly"]
        return wx, wy

    # ── loss terms ──

    def wirelength(self, gamma: float = 5.0) -> torch.Tensor:
        """Log-Sum-Exp smooth HPWL.

        For a net *e* with pin x-coordinates {x_i}:

            HPWL_x ≈ γ·log Σ exp(x_i/γ)  +  γ·log Σ exp(−x_i/γ)

        which converges to  max(x) − min(x)  as γ → 0.
        """
        total = torch.tensor(0.0, device=self.p.device)
        for net in self.p.nets:
            wx, wy = self._pin_world(net)
            w = 0.05 if net["power"] else 1.0
            wl = gamma * (
                torch.logsumexp(wx / gamma, 0)
                + torch.logsumexp(-wx / gamma, 0)
                + torch.logsumexp(wy / gamma, 0)
                + torch.logsumexp(-wy / gamma, 0)
            )
            total = total + w * wl
        return total

    def overlap(self, slop: float = 0.3) -> torch.Tensor:
        """Pairwise AABB overlap area with rotation-aware bounding boxes.

        For a rectangle (hx, hy) rotated by θ the axis-aligned half-extents are:

            Hx = |hx·cosθ| + |hy·sinθ|
            Hy = |hx·sinθ| + |hy·cosθ|
        """
        ct = torch.cos(self.Theta).abs()
        st = torch.sin(self.Theta).abs()
        hx = self.p.half_sizes[:, 0] * ct + self.p.half_sizes[:, 1] * st
        hy = self.p.half_sizes[:, 0] * st + self.p.half_sizes[:, 1] * ct

        # Pairwise separations (N, N)
        dx = (self.X.unsqueeze(1) - self.X.unsqueeze(0)).abs()
        dy = (self.Y.unsqueeze(1) - self.Y.unsqueeze(0)).abs()

        # Required clearance
        sx = hx.unsqueeze(1) + hx.unsqueeze(0) + slop
        sy = hy.unsqueeze(1) + hy.unsqueeze(0) + slop

        area = torch.relu(sx - dx) * torch.relu(sy - dy)
        return area[self.p._triu].sum()

    def boundary(self) -> torch.Tensor:
        """Quadratic penalty for component AABBs outside the board bbox."""
        hx = self.p.half_sizes[:, 0]
        hy = self.p.half_sizes[:, 1]
        bmin = self.p.board_min
        bmax = self.p.board_max

        v = (
            torch.relu(bmin[0] - (self.X - hx)) ** 2
            + torch.relu((self.X + hx) - bmax[0]) ** 2
            + torch.relu(bmin[1] - (self.Y - hy)) ** 2
            + torch.relu((self.Y + hy) - bmax[1]) ** 2
        )
        return v[self.p.is_free].sum()

    def orient(self) -> torch.Tensor:
        """Periodic penalty with minima at 0°, 90°, 180°, 270°.

        sin²(2θ) = 0  when  θ ∈ {0, π/2, π, 3π/2}.
        """
        return (torch.sin(2 * self.Theta[self.p.is_free]) ** 2).sum()

    def decoup_loss(self) -> torch.Tensor:
        """Proximity springs pulling bypass caps toward their target IC."""
        if not self.p.decoup:
            return torch.tensor(0.0, device=self.p.device)

        ci = torch.tensor(
            [d[0] for d in self.p.decoup], dtype=torch.long, device=self.p.device
        )
        ii = torch.tensor(
            [d[1] for d in self.p.decoup], dtype=torch.long, device=self.p.device
        )
        dist = torch.sqrt(
            (self.X[ci] - self.X[ii]) ** 2 + (self.Y[ci] - self.Y[ii]) ** 2 + 1e-8
        )
        target = (
            self.p.half_sizes[ci].sum(1) + self.p.half_sizes[ii].sum(1)
        ) * 0.7
        return torch.relu(dist - target).pow(2).sum()

    # ── combined loss ──

    def forward(self, w: dict) -> tuple[torch.Tensor, dict]:
        L_w = self.wirelength(gamma=w.get("gamma", 5.0))
        L_o = self.overlap()
        L_b = self.boundary()
        L_r = self.orient()
        L_d = self.decoup_loss()

        total = (
            w["wl"] * L_w
            + w["ov"] * L_o
            + w["bd"] * L_b
            + w["or"] * L_r
            + w["dc"] * L_d
        )
        return total, {
            "total": total.item(),
            "wire": L_w.item(),
            "overlap": L_o.item(),
            "boundary": L_b.item(),
            "orient": L_r.item(),
            "decoup": L_d.item(),
        }


# ──────────────────────────────────────────────────────────────────────
# Gradient helpers
# ──────────────────────────────────────────────────────────────────────


def _zero_fixed(model: PlacerModel) -> None:
    """Zero out gradients for fixed (non-movable) components."""
    for attr in ("X", "Y", "Theta"):
        g = getattr(model, attr).grad
        if g is not None:
            g[~model.p.is_free] = 0.0


# ──────────────────────────────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────────────────────────────


def run_gpu_placement(
    components: dict,
    board: np.ndarray,
    *,
    n_iters: int = 1500,
    lr: float = 0.05,
    device: str = "cuda",
    verbose: bool = True,
) -> dict:
    """Run the three-phase GPU-accelerated placement pipeline.

    Mutates *components* in-place (updates ``.pos`` and ``.theta``).
    Returns the final metrics dict.
    """
    if not HAS_TORCH:
        raise ImportError("PyTorch required.  pip install torch")

    t0 = time.perf_counter()

    prob = PlacementProblem(components, board, device)
    model = PlacerModel(prob, components)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, n_iters, eta_min=lr * 0.01
    )

    n_signal = sum(1 for n in prob.nets if not n["power"])
    n_power = len(prob.nets) - n_signal
    if verbose:
        print(
            f"\n  Components: {prob.n}  "
            f"({int(prob.is_free.sum().item())} free, "
            f"{int((~prob.is_free).sum().item())} fixed)"
        )
        print(f"  Nets: {len(prob.nets)} ({n_signal} signal, {n_power} power)")
        print(f"  Decoupling pairs: {len(prob.decoup)}")

    # ── Phase 1: Global Placement ──────────────────────────────────
    if verbose:
        print("\n--- Phase 1: Global Placement ---")

    for it in range(n_iters):
        p = it / n_iters

        # Annealing schedule
        w = {
            "gamma": max(1.0, 10.0 * (1 - p)),  # tighten LSE approximation
            "wl": 1.0,
            "ov": 10.0 + 300.0 * p**2,  # ramp up overlap penalty
            "bd": 50.0 + 300.0 * p,  # ramp up boundary penalty
            "or": 8.0 * max(0.0, p - 0.4) ** 2,  # late-phase orient snap
            "dc": 10.0,
        }

        opt.zero_grad()
        loss, m = model(w)
        loss.backward()
        _zero_fixed(model)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 50.0)
        opt.step()
        sched.step()

        if verbose and (it % 200 == 0 or it == n_iters - 1):
            print(
                f"  [{it:5d}/{n_iters}]  total={m['total']:10.1f}  "
                f"wire={m['wire']:8.1f}  ovlp={m['overlap']:8.1f}  "
                f"bnd={m['boundary']:7.1f}  ori={m['orient']:5.2f}"
            )

    # ── Phase 2: Legalization ──────────────────────────────────────
    if verbose:
        print("\n--- Phase 2: Legalization (snap to 90 deg) ---")

    with torch.no_grad():
        allowed = torch.tensor(
            [0.0, np.pi / 2, np.pi, 3 * np.pi / 2], device=prob.device
        )
        for i in range(prob.n):
            if not prob.is_free[i]:
                continue
            tv = model.Theta[i] % (2 * np.pi)
            diffs = ((allowed - tv + np.pi) % (2 * np.pi)) - np.pi
            model.Theta[i] = allowed[diffs.abs().argmin()]

    # ── Phase 3: Refinement (frozen θ, aggressive overlap) ────────
    if verbose:
        print("\n--- Phase 3: Refinement (frozen theta) ---")

    model.Theta.requires_grad_(False)
    opt2 = torch.optim.AdamW([model.X, model.Y], lr=lr * 0.3)

    refine_iters = 400
    for it in range(refine_iters):
        w = {
            "gamma": 1.0,
            "wl": 1.0,
            "ov": 500.0,
            "bd": 500.0,
            "or": 0.0,
            "dc": 15.0,
        }
        opt2.zero_grad()
        loss, m = model(w)
        loss.backward()
        _zero_fixed(model)
        torch.nn.utils.clip_grad_norm_([model.X, model.Y], 30.0)
        opt2.step()

        if verbose and (it % 100 == 0 or it == refine_iters - 1):
            print(
                f"  [{it:4d}/{refine_iters}]  "
                f"ovlp={m['overlap']:8.1f}  wire={m['wire']:8.1f}  "
                f"bnd={m['boundary']:7.1f}"
            )

    model.Theta.requires_grad_(True)

    # ── Write optimized positions back to Component objects ────────
    with torch.no_grad():
        for i, cid in enumerate(prob.all_ids):
            if prob.is_free[i]:
                components[cid].pos = np.array(
                    [model.X[i].item(), model.Y[i].item()]
                )
                components[cid].theta = model.Theta[i].item()

    dt = time.perf_counter() - t0
    if verbose:
        print(
            f"\n[OK] Placement complete in {dt:.2f}s  "
            f"({prob.n} components, {len(prob.nets)} nets)"
        )

    return m
