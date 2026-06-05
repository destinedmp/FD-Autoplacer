"""
main.py — Entry point for the FD-Autoplacer.

Usage:
    python main.py <board.kicad_pcb> [--export output.kicad_pcb] [--cpu]

Flow:
    1. Parse the .kicad_pcb file  →  dict of Component objects + board polygon
    2. Scramble component positions (so we can see the before/after)
    3. Run the optimizer  →  components are updated in-place
    4. Draw before/after visualization  →  placement.png
    5. Optionally export the placed PCB back to a .kicad_pcb file
"""

from __future__ import annotations
import os
import sys
import argparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from parser import (
    QUAD, Component, Pin, SwapGroup,
    rot, pin_world, rebuild_nets,
    point_in_polygon, closest_on_polygon,
    import_kicad_pcb,
)

# Power net names whose ratsnest lines are hidden in the visualization
# (they connect to almost everything and make the plot unreadable).
POWER_NETS = {"GND", "VCC", "+3.3V", "+5V", "VBUS"}


# ── Visualization ─────────────────────────────────────────────────────


def draw(components: dict, board: np.ndarray, ax, title: str, metrics: dict = None) -> None:
    """Draw the board outline, ratsnest lines, and component boxes."""
    ax.set_aspect("equal")
    ax.set_title(title)

    # Board outline
    ax.add_patch(patches.Polygon(
        board, facecolor="#f6f6e8", edgecolor="#444", linewidth=1.6, zorder=0
    ))

    # Ratsnest — one line from each pin to its net centroid
    for net_name, members in rebuild_nets(components).items():
        if len(members) < 2 or net_name in POWER_NETS:
            continue
        pts = [pin_world(components[cid], components[cid].pins[pi]) for cid, pi in members]
        ctr = np.mean(pts, axis=0)
        for p in pts:
            ax.plot([p[0], ctr[0]], [p[1], ctr[1]], "-", color="#6b6", alpha=0.5, linewidth=0.8)

    # Components
    for cid, c in components.items():
        box = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * c.half_size
        corners = (rot(c.theta) @ box.T).T + c.pos

        is_bottom = c.layer == "B.Cu"
        if is_bottom:
            face, edge, ls = "#fdd", "#822", "--"   # bottom: red dashed
        elif cid.startswith("J"):
            face, edge, ls = "#fcc", "#225", "-"    # connectors: salmon
        else:
            face, edge, ls = "#cce", "#225", "-"    # regular: blue

        hatch = "////" if c.fixed else None

        ax.add_patch(patches.Polygon(corners, facecolor=face, edgecolor=edge,
                                     linewidth=1.0, linestyle=ls, hatch=hatch, zorder=2))
        ax.text(c.pos[0], c.pos[1], cid, ha="center", va="center",
                fontsize=6.5, color="#822" if is_bottom else "#000", zorder=3)
        for p in c.pins:
            wp = pin_world(c, p)
            ax.plot(wp[0], wp[1], "o", color="#822" if is_bottom else "#225",
                    markersize=1.8, zorder=3)

    # Legend
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#cce", edgecolor="#225", linestyle="-", label="Top Layer (F.Cu)"),
        Patch(facecolor="#fdd", edgecolor="#822", linestyle="--", label="Bottom Layer (B.Cu)"),
        Patch(facecolor="#fcc", edgecolor="#225", linestyle="-", label="Connector"),
        Patch(facecolor="white", edgecolor="#444", hatch="////", label="Locked/Fixed")
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=8)

    # Metrics overlay
    if metrics is not None:
        textstr = "\\n".join((
            f"Components: {metrics.get('n_total', 0)} ({metrics.get('n_top', 0)} Top, {metrics.get('n_bot', 0)} Bot)",
            f"Time: {metrics.get('time', 0.0):.2f}s",
            f"Wirelength Loss: {metrics.get('wire', 0.0):.1f}",
            f"Overlap Loss: {metrics.get('overlap', 0.0):.1f}",
            f"Boundary Loss: {metrics.get('boundary', 0.0):.1f}"
        ))
        props = dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="#ccc")
        ax.text(0.02, 0.02, textstr, transform=ax.transAxes, fontsize=8,
                verticalalignment="bottom", bbox=props)


# ── Boundary projection (used after GPU placement) ────────────────────


def project_position(comp: Component) -> None:
    """
    Push a component back inside the board polygon if it has drifted outside.
    The GPU optimizer uses a soft boundary penalty during training; this does
    a final hard correction pass afterwards.

    Loops up to 6 times because one pass may push the centre back in but leave
    corners still outside — iterating converges to a fully-contained position.
    """
    poly = comp.allowed_polygon
    if poly is None:
        return
    if not point_in_polygon(comp.pos, poly):
        comp.pos = closest_on_polygon(comp.pos, poly)
    hs = comp.half_size
    for _ in range(6):
        corners = comp.pos + np.array([[-hs[0], -hs[1]], [hs[0], -hs[1]],
                                        [hs[0], hs[1]], [-hs[0], hs[1]]])
        push = np.zeros(2)
        moved = False
        for corner in corners:
            if not point_in_polygon(corner, poly):
                d = closest_on_polygon(corner, poly) - corner
                if abs(d[0]) > abs(push[0]):
                    push[0] = d[0]
                if abs(d[1]) > abs(push[1]):
                    push[1] = d[1]
                moved = True
        if not moved:
            break
        comp.pos = comp.pos + push


# ── Pin swap (Hungarian algorithm) ───────────────────────────────────


def optimize_swap_group(components: dict, nets: dict, group: SwapGroup) -> None:
    """
    For symmetric components (e.g. a resistor whose two pins are interchangeable),
    find the net assignment that minimises total wirelength using the Hungarian algorithm.
    """
    from scipy.optimize import linear_sum_assignment
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


# ── CLI ────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Force-Directed PCB Autoplacer (gradient-descent / GPU)"
    )
    ap.add_argument("pcb", nargs="?", default=os.path.join("tests", "simple.kicad_pcb"),
                    help="Path to .kicad_pcb file")
    ap.add_argument("--cpu", action="store_true",
                    help="Force CPU even when CUDA is available")
    ap.add_argument("--iters", type=int, default=1500,
                    help="Global-placement iterations (default: 1500)")
    ap.add_argument("--lr", type=float, default=0.05,
                    help="AdamW learning rate (default: 0.05)")
    ap.add_argument("--export", type=str, default=None,
                    help="Write placed PCB to this .kicad_pcb path")
    ap.add_argument("--no-scramble", action="store_true",
                    help="Keep original positions instead of randomising")
    args = ap.parse_args()

    # ── Load board ──
    if not os.path.exists(args.pcb):
        print(f"[!] PCB file not found: {args.pcb}")
        sys.exit(1)

    components, board = import_kicad_pcb(args.pcb)

    # Fallback board outline if Edge.Cuts is missing
    if board is None:
        pts = np.array([c.pos for c in components.values()])
        lo, hi = pts.min(axis=0) - 20, pts.max(axis=0) + 20
        board = np.array([[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]])
        for c in components.values():
            c.allowed_polygon = board

    # Auto-detect all symmetric 2-pin passives (resistors, caps, diodes,
    # ferrite beads, inductors) — their two pins are physically interchangeable.
    SYMMETRIC_PREFIXES = ("R", "C", "D", "FB", "L", "F")
    swap_groups = [
        SwapGroup(cid, [0, 1])
        for cid, comp in components.items()
        if cid.upper().startswith(SYMMETRIC_PREFIXES)
        and len(comp.pins) == 2
        and not comp.fixed
    ]
    print(f"  Pin-swap groups: {len(swap_groups)} symmetric 2-pin passives")

    # ── Scramble starting positions ──
    xs, ys = board[:, 0], board[:, 1]
    if not args.no_scramble:
        rng = np.random.default_rng(0)
        for c in components.values():
            if c.id.startswith("J"):
                c.fixed = True
                continue
            c.pos = rng.uniform([xs.min() + 2, ys.min() + 2],
                                [xs.max() - 2, ys.max() - 2])
            project_position(c)

    # ── Before snapshot ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    draw(components, board, axes[0], "Initial (Scrambled)")

    # ── Run optimizer ──
    from optim_engine import run_gpu_placement
    device = "cpu" if args.cpu else "cuda"
    metrics = run_gpu_placement(components, board, n_iters=args.iters, lr=args.lr, device=device)
    # Hard-project any components that drifted just outside the polygon
    for comp in components.values():
        if not comp.fixed:
            project_position(comp)
    # Discrete pin-swap pass (not handled by the continuous optimizer)
    nets = rebuild_nets(components)
    for g in swap_groups:
        optimize_swap_group(components, nets, g)
    title = "Converged (GPU Optimized)"

    # ── After snapshot + save ──
    draw(components, board, axes[1], title, metrics=metrics)
    for ax in axes:
        ax.set_xlim(xs.min() - 3, xs.max() + 3)
        ax.set_ylim(ys.min() - 3, ys.max() + 3)
        ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig("placement.png", dpi=110)
    print("Saved layout visualization to placement.png")

    # ── Export ──
    if args.export:
        from kicad_exporter import export_kicad_pcb
        export_kicad_pcb(args.pcb, args.export, components)

    print("Placement optimization done.")
