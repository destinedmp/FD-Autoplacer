"""
KiCad PCB file exporter.

Reads an existing ``.kicad_pcb`` file, updates component ``(at X Y θ)``
coordinates from the optimizer output, and writes a new file.  Uses AST-level
manipulation so that all non-positional data (tracks, zones, text, etc.) is
preserved verbatim.
"""

from __future__ import annotations

import numpy as np


# ──────────────────────────────────────────────────────────────────────
# S-expression serializer
# ──────────────────────────────────────────────────────────────────────


def _format_sexp(node, depth: int = 0) -> str:
    """Serialize an S-expression AST back to a formatted string.

    Leaf strings are returned as-is.  Lists whose children are all atoms
    are rendered on a single line ``(tag a b c)``.  Lists containing nested
    lists are rendered with KiCad-style indentation.
    """
    if isinstance(node, str):
        return node

    # All-atom node → compact single line
    if all(isinstance(c, str) for c in node):
        return "(" + " ".join(node) + ")"

    indent = "  " * depth
    child_indent = "  " * (depth + 1)

    # Header: opening paren + tag + any leading atom children
    header = "("
    rest_start = 0
    for j, child in enumerate(node):
        if isinstance(child, str):
            header += (" " if j > 0 else "") + child
            rest_start = j + 1
        else:
            break

    lines = [header]
    for child in node[rest_start:]:
        if isinstance(child, list):
            lines.append(child_indent + _format_sexp(child, depth + 1))
        else:
            lines.append(child_indent + child)
    lines.append(indent + ")")
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────


def export_kicad_pcb(
    input_path: str,
    output_path: str,
    components: dict,
) -> None:
    """Write a new ``.kicad_pcb`` with updated component positions.

    Parameters
    ----------
    input_path : str
        Path to the original ``.kicad_pcb`` file (used as the template).
    output_path : str
        Destination path for the placed file.
    components : dict
        Mapping of reference → Component with optimized ``.pos`` and ``.theta``.
    """
    # Re-use the parser already available in main.py
    from main import parse_sexp, _find, _first, _unq

    root = parse_sexp(open(input_path, encoding="utf-8").read())
    updated = 0

    for fp in _find(root, "footprint"):
        # ── resolve component reference ──
        ref = None
        for t in _find(fp, "fp_text"):
            if len(t) > 2 and t[1] == "reference":
                ref = _unq(t[2])
                break
        if ref is None:
            for p in _find(fp, "property"):
                if len(p) > 2 and _unq(p[1]) == "Reference":
                    ref = _unq(p[2])
                    break
        if ref is None or ref not in components:
            continue

        comp = components[ref]
        at = _first(fp, "at")
        if at is None:
            continue

        # Convert back to KiCad coordinate frame (Y-down, degrees)
        new_x = comp.pos[0]
        new_y = -comp.pos[1]  # negate Y back to KiCad Y-down
        angle_deg = float(np.degrees(comp.theta)) % 360

        at[1] = f"{new_x:.4f}"
        at[2] = f"{new_y:.4f}"
        if len(at) > 3:
            at[3] = f"{angle_deg:.2f}"
        elif abs(angle_deg) > 0.01:
            at.append(f"{angle_deg:.2f}")

        updated += 1

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(_format_sexp(root))
        f.write("\n")

    print(f"[OK] Exported {updated} component positions -> {output_path}")
