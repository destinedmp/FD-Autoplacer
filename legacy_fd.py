"""
legacy_fd.py — Classic force-directed PCB placement solver.

This is the original Python-loop implementation kept for reference and
as a fallback when PyTorch is not available. It is NOT called by default.
Run with: python main.py <board.kicad_pcb> --legacy

How it works:
    Each iteration applies two forces to every component:
      - Attractive: pins connected to the same net pull toward their centroid
      - Repulsive:  overlapping component bounding boxes push apart
    Position and rotation are updated via simple Euler integration:
        pos += dt * (F_attractive + F_repulsive)
        theta += dt * torque
    Rotations are snapped to 90° multiples at the end.
    Every 25 steps a Hungarian-algorithm pin-swap pass is run on
    symmetric components (resistors, etc.) to reduce wirelength.
"""

from __future__ import annotations
import numpy as np
from scipy.optimize import linear_sum_assignment

from parser import (
    Component, SwapGroup,
    rot, pin_world, rebuild_nets,
    point_in_polygon, closest_on_polygon,
)


# ── Geometry helper (only needed internally) ──────────────────────────


def _hs_rot(c: Component) -> np.ndarray:
    """Effective AABB half-size after snapping rotation to nearest 90°."""
    return c.half_size[::-1] if round(c.theta / (np.pi / 2)) % 2 else c.half_size


# ── Forces ────────────────────────────────────────────────────────────


def attractive(components: dict, nets: dict) -> tuple[dict, dict]:
    """
    Pull each pin toward its net centroid.
    Returns (translational_forces, torques) per component.
    """
    F_lin = {cid: np.zeros(2) for cid in components}
    F_tor = {cid: 0.0 for cid in components}
    for members in nets.values():
        if len(members) < 2:
            continue
        positions = [pin_world(components[cid], components[cid].pins[pi]) for cid, pi in members]
        center = np.mean(positions, axis=0)
        for (cid, _), pos in zip(members, positions):
            f = center - pos
            F_lin[cid] += f
            r = pos - components[cid].pos
            F_tor[cid] += r[0] * f[1] - r[1] * f[0]   # 2D cross product = torque
    return F_lin, F_tor


def repulsive(components: dict, k: float = 30.0, slop: float = 0.4) -> dict:
    """
    Push overlapping components apart along the axis of least penetration.
    k controls how hard the push is; slop adds a small clearance gap.
    """
    F = {cid: np.zeros(2) for cid in components}
    ids = list(components.keys())
    for i in range(len(ids)):
        a = components[ids[i]]
        ah = _hs_rot(a)
        for j in range(i + 1, len(ids)):
            b = components[ids[j]]
            bh = _hs_rot(b)
            dx, dy = b.pos[0] - a.pos[0], b.pos[1] - a.pos[1]
            ox = ah[0] + bh[0] + slop - abs(dx)
            oy = ah[1] + bh[1] + slop - abs(dy)
            if ox <= 0 or oy <= 0:
                continue    # no overlap
            if ox < oy:
                push = np.array([np.sign(dx or 1) * ox * k, 0.0])
            else:
                push = np.array([0.0, np.sign(dy or 1) * oy * k])
            F[ids[i]] -= push
            F[ids[j]] += push
    return F


# ── Constraints ───────────────────────────────────────────────────────


def project_position(comp: Component) -> None:
    """Project component centre (and body corners) back inside the board polygon."""
    poly = comp.allowed_polygon
    if poly is None:
        return
    if not point_in_polygon(comp.pos, poly):
        comp.pos = closest_on_polygon(comp.pos, poly)
    for _ in range(6):
        hs = _hs_rot(comp)
        corners = comp.pos + np.array([[-hs[0], -hs[1]], [hs[0], -hs[1]],
                                        [hs[0], hs[1]], [-hs[0], hs[1]]])
        push = np.zeros(2)
        moved = False
        for corner in corners:
            if point_in_polygon(corner, poly):
                continue
            d = closest_on_polygon(corner, poly) - corner
            if abs(d[0]) > abs(push[0]):
                push[0] = d[0]
            if abs(d[1]) > abs(push[1]):
                push[1] = d[1]
            moved = True
        if not moved:
            return
        comp.pos = comp.pos + push


def snap_orientation(comp: Component, strength: float) -> None:
    """Blend rotation toward nearest allowed angle. strength=1.0 is a hard snap."""
    if comp.allowed_orientations is None or strength <= 0:
        return
    deltas = [(a - comp.theta + np.pi) % (2 * np.pi) - np.pi for a in comp.allowed_orientations]
    comp.theta += strength * min(deltas, key=abs)


# ── Pin swap ──────────────────────────────────────────────────────────


def optimize_swap_group(components: dict, nets: dict, group: SwapGroup) -> None:
    """Re-assign nets on interchangeable pins to minimise wirelength (Hungarian algorithm)."""
    comp = components[group.component_id]
    pins = [comp.pins[i] for i in group.pin_indices]
    nets_g = [p.net for p in pins]
    pos_g = [pin_world(comp, p) for p in pins]
    centroids = []
    for net in nets_g:
        others = [
            pin_world(components[cid], components[cid].pins[pi])
            for cid, pi in nets[net]
            if not (cid == comp.id and pi in group.pin_indices)
        ]
        centroids.append(np.mean(others, axis=0) if others else comp.pos)
    n = len(pins)
    cost = np.array([[float(np.sum((pos_g[i] - centroids[j]) ** 2))
                      for j in range(n)] for i in range(n)])
    _, col = linear_sum_assignment(cost)
    for i, j in enumerate(col):
        comp.pins[group.pin_indices[i]].net = nets_g[j]


# ── Main solver loop ──────────────────────────────────────────────────


def run(components: dict, swap_groups: list, n_iters: int = 800,
        dt0: float = 0.05, swap_every: int = 25, snap_phase: float = 0.4) -> None:
    """
    Run the force-directed placement for n_iters steps.
    dt decays each iteration so the system settles rather than oscillates.
    Orientation snapping is gradually introduced after snap_phase fraction of iters.
    """
    dt = dt0
    for it in range(n_iters):
        nets = rebuild_nets(components)
        Fa, Tor = attractive(components, nets)
        Fr = repulsive(components)
        snap_str = max(0.0, (it / n_iters - snap_phase) / (1 - snap_phase)) ** 2

        for cid, comp in components.items():
            if comp.fixed:
                continue
            comp.pos = comp.pos + dt * (Fa[cid] + Fr[cid])
            if comp.allowed_orientations and len(comp.allowed_orientations) == 1:
                comp.theta = comp.allowed_orientations[0]
            else:
                comp.theta += dt * 0.15 * Tor[cid]
                snap_orientation(comp, snap_str * 0.5)
            project_position(comp)

        if it > 0 and it % swap_every == 0:
            for g in swap_groups:
                optimize_swap_group(components, nets, g)
        dt *= 0.996

    # Final hard snap + one last pin-swap pass
    for comp in components.values():
        if not comp.fixed:
            snap_orientation(comp, 1.0)
            project_position(comp)
    nets = rebuild_nets(components)
    for g in swap_groups:
        optimize_swap_group(components, nets, g)
