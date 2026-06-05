"""
parser.py — KiCad PCB file parser and shared data models.

Reads a .kicad_pcb file and returns a dict of Component objects
plus a NumPy polygon representing the board boundary.

KiCad uses S-expressions (Lisp-style nested lists) and a Y-down
coordinate frame. We negate Y on read so everything internal uses
Y-up, which matches matplotlib.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

# The four cardinal orientations allowed for most components (radians).
QUAD = [0.0, np.pi / 2, np.pi, 3 * np.pi / 2]


# ── Data models ──────────────────────────────────────────────────────


@dataclass
class Pin:
    """A single pad on a component, in component-local coordinates."""
    local_pos: np.ndarray   # (x, y) relative to component centre
    net: str                # net name this pad connects to


@dataclass
class Component:
    """One footprint on the board."""
    id: str                                # reference designator, e.g. "R3"
    pos: np.ndarray                        # (x, y) world position
    theta: float = 0.0                     # rotation in radians
    half_size: np.ndarray = field(default_factory=lambda: np.array([2.5, 2.5]))
    pins: list = field(default_factory=list)
    fixed: bool = False                    # True = connector pinned to edge
    layer: str = "F.Cu"                   # "F.Cu" (top) or "B.Cu" (bottom)
    allowed_rect: tuple | None = None      # hard rectangular constraint
    allowed_polygon: np.ndarray | None = None  # board outline polygon
    allowed_orientations: list | None = None   # discrete angles allowed


@dataclass
class SwapGroup:
    """A set of interchangeable pins on one component (e.g. both ends of a resistor)."""
    component_id: str
    pin_indices: list


# ── Geometry helpers ─────────────────────────────────────────────────


def rot(theta: float) -> np.ndarray:
    """2×2 rotation matrix for angle theta (radians)."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def pin_world(comp: Component, pin: Pin) -> np.ndarray:
    """Convert a pin's local position to world coordinates."""
    return comp.pos + rot(comp.theta) @ pin.local_pos


def rebuild_nets(components: dict) -> dict:
    """Build a net → [(component_id, pin_index)] mapping from all components."""
    nets: dict = {}
    for cid, comp in components.items():
        for pi, pin in enumerate(comp.pins):
            nets.setdefault(pin.net, []).append((cid, pi))
    return nets


def point_in_polygon(p: np.ndarray, poly: np.ndarray) -> bool:
    """Ray-casting point-in-polygon test."""
    inside, n, j = False, len(poly), len(poly) - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > p[1]) != (yj > p[1])) and (
            p[0] < (xj - xi) * (p[1] - yi) / (yj - yi + 1e-12) + xi
        ):
            inside = not inside
        j = i
    return inside


def closest_on_polygon(p: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Return the closest point on any polygon edge to point p."""
    best, best_d = poly[0], np.inf
    for i in range(len(poly)):
        a, b = poly[i], poly[(i + 1) % len(poly)]
        ab = b - a
        t = max(0.0, min(1.0, float((p - a) @ ab) / (float(ab @ ab) + 1e-12)))
        q = a + t * ab
        d = float((p - q) @ (p - q))
        if d < best_d:
            best_d, best = d, q
    return best


# ── S-expression parser ───────────────────────────────────────────────
# KiCad files look like: (kicad_pcb (version 20221018) (footprint "R_0402" ...))
# parse_sexp turns that text into nested Python lists.


def parse_sexp(text: str):
    """Tokenise and parse a KiCad S-expression string into nested Python lists."""
    tokens, i, n = [], 0, len(text)
    while i < n:
        c = text[i]
        if c.isspace():
            i += 1
        elif c in "()":
            tokens.append(c)
            i += 1
        elif c == '"':
            j = i + 1
            while j < n and text[j] != '"':
                j += 2 if text[j] == "\\" else 1
            tokens.append(text[i: j + 1])
            i = j + 1
        else:
            j = i
            while j < n and not text[j].isspace() and text[j] not in "()":
                j += 1
            tokens.append(text[i:j])
            i = j

    p = [0]

    def parse():
        if tokens[p[0]] != "(":
            t = tokens[p[0]]
            p[0] += 1
            return t
        p[0] += 1
        out = []
        while tokens[p[0]] != ")":
            out.append(parse())
        p[0] += 1
        return out

    return parse()


def _find(node, tag):
    """Return all direct child nodes with the given tag name."""
    return [c for c in node if isinstance(c, list) and c and c[0] == tag]


def _first(node, tag):
    """Return the first direct child node with the given tag, or None."""
    for c in node:
        if isinstance(c, list) and c and c[0] == tag:
            return c
    return None


def _unq(s):
    """Strip surrounding quotes from a KiCad string token."""
    return s[1:-1] if isinstance(s, str) and len(s) >= 2 and s[0] == '"' == s[-1] else s


def _chain_segments(segs: list, tol: float = 0.01) -> np.ndarray | None:
    """
    Join a soup of line segments into a single closed polygon.

    KiCad stores board outlines as individual gr_line / gr_rect segments,
    not as a single polyline. This function stitches them together by
    matching endpoints within `tol` mm.
    """
    if not segs:
        return None
    k = lambda p: (round(p[0] / tol) * tol, round(p[1] / tol) * tol)
    rem = list(segs)
    poly = [rem[0][0], rem[0][1]]
    rem.pop(0)
    while rem:
        last = k(poly[-1])
        for i, (a, b) in enumerate(rem):
            if k(a) == last:
                poly.append(b)
                rem.pop(i)
                break
            if k(b) == last:
                poly.append(a)
                rem.pop(i)
                break
        else:
            break
    if len(poly) > 1 and k(poly[0]) == k(poly[-1]):
        poly.pop()
    return np.array(poly)


# ── Main import function ──────────────────────────────────────────────


def import_kicad_pcb(path: str) -> tuple[dict, np.ndarray | None]:
    """
    Parse a .kicad_pcb file and return (components, board_polygon).

    components : dict[str, Component]
        Keyed by reference designator (e.g. "R3", "U1", "J1").
    board_polygon : np.ndarray | None
        Vertices of the Edge.Cuts outline in Y-up coordinates.
        None if no board outline is found.
    """
    root = parse_sexp(open(path, encoding="utf-8").read())
    comps: dict = {}

    for fp in _find(root, "footprint"):
        # ── Find reference designator ──
        ref = next(
            (_unq(t[2]) for t in _find(fp, "fp_text") if len(t) > 2 and t[1] == "reference"),
            None,
        )
        if not ref:
            ref = next(
                (_unq(p[2]) for p in _find(fp, "property") if len(p) > 2 and _unq(p[1]) == "Reference"),
                None,
            )
        if not ref:
            continue

        # ── Position and rotation ──
        at = _first(fp, "at")
        fx, fy = float(at[1]), -float(at[2])     # negate Y: KiCad Y-down → Y-up
        ftheta = np.radians(float(at[3])) if len(at) > 3 else 0.0

        # ── Layer (F.Cu = top, B.Cu = bottom) ──
        fp_layer_node = _first(fp, "layer")
        fp_layer = _unq(fp_layer_node[1]) if fp_layer_node and len(fp_layer_node) > 1 else "F.Cu"

        # ── Pads → pins + bounding box estimate ──
        pins, xs, ys = [], [], []
        for pad in _find(fp, "pad"):
            pa = _first(pad, "at")
            px, py = float(pa[1]), -float(pa[2])
            nn = _first(pad, "net")
            net = _unq(nn[2] if len(nn) > 2 else nn[1]) if nn else f"NC_{ref}_{_unq(pad[1])}"
            pins.append(Pin(np.array([px, py]), net))
            sz = _first(pad, "size")
            sw, sh = float(sz[1]), float(sz[2])
            xs.extend([px - sw / 2, px + sw / 2])
            ys.extend([py - sh / 2, py + sh / 2])

        # Half-size from pad extents (courtyard layer would be more accurate
        # but is optional in KiCad, so we approximate from pads + 0.3 mm margin)
        hw = max(abs(min(xs)), abs(max(xs))) + 0.3 if xs else 1.5
        hh = max(abs(min(ys)), abs(max(ys))) + 0.3 if ys else 1.5

        comps[ref] = Component(
            id=ref,
            pos=np.array([fx, fy]),
            theta=ftheta,
            half_size=np.array([hw, hh]),
            pins=pins,
            layer=fp_layer,
            allowed_orientations=QUAD,
        )

    # ── Board outline (Edge.Cuts layer) ──
    segs = []
    for gl in _find(root, "gr_line"):
        ly = _first(gl, "layer")
        if ly and _unq(ly[1]) == "Edge.Cuts":
            s, e = _first(gl, "start"), _first(gl, "end")
            segs.append(((float(s[1]), -float(s[2])), (float(e[1]), -float(e[2]))))
    for ga in _find(root, "gr_arc"):
        ly = _first(ga, "layer")
        if ly and _unq(ly[1]) == "Edge.Cuts":
            s, e = _first(ga, "start"), _first(ga, "end")
            if s and e:
                segs.append(((float(s[1]), -float(s[2])), (float(e[1]), -float(e[2]))))
    for gr in _find(root, "gr_rect"):
        ly = _first(gr, "layer")
        if ly and _unq(ly[1]) == "Edge.Cuts":
            s, e = _first(gr, "start"), _first(gr, "end")
            x1, y1 = float(s[1]), -float(s[2])
            x2, y2 = float(e[1]), -float(e[2])
            segs.extend([
                ((x1, y1), (x2, y1)),
                ((x2, y1), (x2, y2)),
                ((x2, y2), (x1, y2)),
                ((x1, y2), (x1, y1)),
            ])

    board = _chain_segments(segs)
    for c in comps.values():
        c.allowed_polygon = board
    return comps, board
