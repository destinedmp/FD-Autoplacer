"""
generate_demo.py — Generates tests/demo_two_layer.kicad_pcb

Circuit: USB-powered microcontroller board
  - U1: STM32-like MCU (large IC, center of board)
  - U2: LDO 3.3V regulator
  - J1: USB-C connector (fixed, left edge)
  - J2: SWD debug header (fixed, right edge)
  - Y1: 16MHz crystal oscillator
  - C1–C4: 100nF MCU decoupling caps  ← optimal placement: B.Cu under U1
  - C5, C6: 10uF bulk caps for LDO
  - C7, C8: 18pF crystal load caps
  - R1: 10k nRESET pull-up
  - R2, R3: 33Ω USB series resistors
  - R4, R5: 100Ω LED current limiters
  - D1, D2: Status LEDs
  - FB1: Ferrite bead on VBUS

All components start on F.Cu.  The optimizer should discover that
placing C1–C4 (and possibly other passives) on B.Cu directly under
the MCU reduces wirelength to the MCU power pins.

Run with:
    python tests/generate_demo.py
    python main.py tests/demo_two_layer.kicad_pcb --export tests/demo_two_layer_placed.kicad_pcb
"""

from __future__ import annotations

# ── Net table ─────────────────────────────────────────────────────────

NETS = {
    0:  "",
    1:  "GND",
    2:  "VBUS",
    3:  "VCC",
    4:  "USB_DP",
    5:  "USB_DM",
    6:  "XTAL_IN",
    7:  "XTAL_OUT",
    8:  "LED1_PWM",
    9:  "LED2_PWM",
    10: "SWDIO",
    11: "SWDCLK",
    12: "nRESET",
    13: "MCU_DP",
    14: "MCU_DM",
    15: "LED1_A",
    16: "LED2_A",
    17: "VBUS_F",
}

# ── S-expression helpers ──────────────────────────────────────────────


def _net(n: int) -> str:
    return f'(net {n} "{NETS[n]}")'


def _pad(num: str, kind: str, shape: str, at: tuple, size: tuple,
         net: int, layers: str = '"F.Cu" "F.Mask"') -> str:
    at_str = f"{at[0]:.3f} {at[1]:.3f}"
    sz_str = f"{size[0]:.3f} {size[1]:.3f}"
    if kind == "thru_hole":
        drill = min(size) * 0.6
        return (f'    (pad "{num}" thru_hole {shape} '
                f'(at {at_str}) (size {sz_str}) (drill {drill:.2f}) '
                f'(layers "*.Cu" "*.Mask") {_net(net)})')
    return (f'    (pad "{num}" {kind} {shape} '
            f'(at {at_str}) (size {sz_str}) '
            f'(layers {layers}) {_net(net)})')


def _fp_text(ref: str, val: str) -> list[str]:
    return [
        f'    (fp_text reference "{ref}" (at 0 0) (layer "F.SilkS") '
        f'(effects (font (size 0.8 0.8))))',
        f'    (fp_text value "{val}" (at 0 1) (layer "F.Fab"))',
    ]


def footprint(ref: str, val: str, kicad_x: float, kicad_y: float,
              pads: list[str], layer: str = "F.Cu", theta: float = 0) -> list[str]:
    """Build a footprint S-expression block."""
    at = f"{kicad_x:.3f} {kicad_y:.3f}" + (f" {theta:.1f}" if theta else "")
    lines = [
        f'  (footprint "Demo:{ref}" (layer "{layer}")',
        f'    (at {at})',
    ]
    lines += _fp_text(ref, val)
    lines += pads
    lines.append("  )")
    return lines


# ── Component definitions ─────────────────────────────────────────────


def u1_mcu() -> list[str]:
    """
    STM32-like MCU — QFP-like, 6×6 mm body.
    Pins on all four sides so decoupling caps are useful on B.Cu.
    Initial position: board centre (15, 11).
    """
    pads = [
        # Left edge pins  (KiCad: x=-3, y increases downward)
        _pad("1",  "smd", "rect", (-3.0, -2.5), (0.6, 0.6), 3),   # VCC
        _pad("2",  "smd", "rect", (-3.0, -1.5), (0.6, 0.6), 1),   # GND
        _pad("3",  "smd", "rect", (-3.0, -0.5), (0.6, 0.6), 13),  # MCU_DP
        _pad("4",  "smd", "rect", (-3.0,  0.5), (0.6, 0.6), 14),  # MCU_DM
        _pad("5",  "smd", "rect", (-3.0,  1.5), (0.6, 0.6), 6),   # XTAL_IN
        _pad("6",  "smd", "rect", (-3.0,  2.5), (0.6, 0.6), 7),   # XTAL_OUT
        # Bottom edge pins
        _pad("7",  "smd", "rect", (-2.0,  3.0), (0.6, 0.6), 8),   # LED1_PWM
        _pad("8",  "smd", "rect", (-1.0,  3.0), (0.6, 0.6), 9),   # LED2_PWM
        _pad("9",  "smd", "rect", ( 0.0,  3.0), (0.6, 0.6), 10),  # SWDIO
        _pad("10", "smd", "rect", ( 1.0,  3.0), (0.6, 0.6), 11),  # SWDCLK
        _pad("11", "smd", "rect", ( 2.0,  3.0), (0.6, 0.6), 12),  # nRESET
        # Right edge pins
        _pad("12", "smd", "rect", ( 3.0,  2.0), (0.6, 0.6), 3),   # VCC
        _pad("13", "smd", "rect", ( 3.0,  1.0), (0.6, 0.6), 1),   # GND
        _pad("14", "smd", "rect", ( 3.0,  0.0), (0.6, 0.6), 1),   # GND
        _pad("15", "smd", "rect", ( 3.0, -1.0), (0.6, 0.6), 1),   # GND
    ]
    return footprint("U1", "STM32F042", 15.0, 11.0, pads)


def u2_ldo() -> list[str]:
    """LDO regulator — SOT-223 like, 4×3 mm."""
    pads = [
        _pad("1", "smd", "rect", (-2.3, -1.5), (1.2, 1.5), 17),  # VIN (VBUS_F)
        _pad("2", "smd", "rect", ( 0.0, -1.5), (1.2, 1.5), 1),   # GND
        _pad("3", "smd", "rect", ( 2.3, -1.5), (1.2, 1.5), 3),   # VOUT (VCC)
        _pad("4", "smd", "rect", ( 0.0,  1.5), (3.5, 2.5), 1),   # GND tab
    ]
    return footprint("U2", "LDO_3V3", 25.0, 7.0, pads)


def j1_usb() -> list[str]:
    """USB-C connector — fixed at left edge, pads face right into board."""
    pads = [
        _pad("A1",  "smd", "rect", (2.5, -2.0), (0.8, 1.2), 1),   # GND
        _pad("A4",  "smd", "rect", (2.5, -1.0), (0.8, 1.2), 2),   # VBUS
        _pad("A6",  "smd", "rect", (2.5,  0.0), (0.8, 1.2), 4),   # USB_DP
        _pad("A7",  "smd", "rect", (2.5,  1.0), (0.8, 1.2), 5),   # USB_DM
        _pad("A12", "smd", "rect", (2.5,  2.0), (0.8, 1.2), 1),   # GND
    ]
    return footprint("J1", "USB-C", 1.5, 11.0, pads)


def j2_swd() -> list[str]:
    """SWD debug header — fixed at right edge, 5-pin 2.54 mm thru-hole."""
    pads = [
        _pad("1", "thru_hole", "circle", (-1.5, -5.08), (1.7, 1.7), 3),   # VCC
        _pad("2", "thru_hole", "circle", (-1.5, -2.54), (1.7, 1.7), 10),  # SWDIO
        _pad("3", "thru_hole", "circle", (-1.5,  0.00), (1.7, 1.7), 11),  # SWDCLK
        _pad("4", "thru_hole", "circle", (-1.5,  2.54), (1.7, 1.7), 12),  # nRESET
        _pad("5", "thru_hole", "circle", (-1.5,  5.08), (1.7, 1.7), 1),   # GND
    ]
    return footprint("J2", "SWD", 28.5, 11.0, pads)


def y1_crystal() -> list[str]:
    """16 MHz crystal — 5×3.2 mm, 2 pads."""
    pads = [
        _pad("1", "smd", "rect", (-1.5, 0.0), (0.8, 1.6), 6),   # XTAL_IN
        _pad("2", "smd", "rect", ( 1.5, 0.0), (0.8, 1.6), 7),   # XTAL_OUT
    ]
    return footprint("Y1", "16MHz", 11.0, 7.0, pads)


def cap_0402(ref: str, val: str, net_p: int, net_n: int,
             kx: float, ky: float) -> list[str]:
    """0402 capacitor — 1.0×0.5 mm body."""
    pads = [
        _pad("1", "smd", "rect", (-0.85, 0.0), (0.5, 0.5), net_p),
        _pad("2", "smd", "rect", ( 0.85, 0.0), (0.5, 0.5), net_n),
    ]
    return footprint(ref, val, kx, ky, pads)


def cap_0805(ref: str, val: str, net_p: int, net_n: int,
             kx: float, ky: float) -> list[str]:
    """0805 bulk capacitor — 2.0×1.2 mm body."""
    pads = [
        _pad("1", "smd", "rect", (-1.4, 0.0), (1.0, 1.2), net_p),
        _pad("2", "smd", "rect", ( 1.4, 0.0), (1.0, 1.2), net_n),
    ]
    return footprint(ref, val, kx, ky, pads)


def res_0402(ref: str, val: str, net1: int, net2: int,
             kx: float, ky: float) -> list[str]:
    """0402 resistor — 1.0×0.5 mm body."""
    pads = [
        _pad("1", "smd", "rect", (-0.85, 0.0), (0.5, 0.5), net1),
        _pad("2", "smd", "rect", ( 0.85, 0.0), (0.5, 0.5), net2),
    ]
    return footprint(ref, val, kx, ky, pads)


def led_0805(ref: str, val: str, net_a: int, net_k: int,
             kx: float, ky: float) -> list[str]:
    """0805 LED — anode/cathode marked."""
    pads = [
        _pad("A", "smd", "rect", ( 1.4, 0.0), (1.0, 1.2), net_a),
        _pad("K", "smd", "rect", (-1.4, 0.0), (1.0, 1.2), net_k),
    ]
    return footprint(ref, val, kx, ky, pads)


def fb_0402(ref: str, kx: float, ky: float) -> list[str]:
    """Ferrite bead 0402."""
    pads = [
        _pad("1", "smd", "rect", (-0.85, 0.0), (0.5, 0.5), 2),   # VBUS in
        _pad("2", "smd", "rect", ( 0.85, 0.0), (0.5, 0.5), 17),  # VBUS_F out
    ]
    return footprint(ref, "600R@100MHz", kx, ky, pads)


# ── File assembly ─────────────────────────────────────────────────────


def build_pcb() -> str:
    lines: list[str] = []

    # Header
    lines += [
        "(kicad_pcb",
        "  (version 20221018)",
        "  (generator pcbnew)",
        "  (general",
        "    (thickness 1.6)",
        "  )",
        "  (paper \"A4\")",
        "  (layers",
        "    (0 \"F.Cu\" signal)",
        "    (31 \"B.Cu\" signal)",
        "    (36 \"B.SilkS\" user \"B.Silkscreen\")",
        "    (37 \"F.SilkS\" user \"F.Silkscreen\")",
        "    (38 \"B.Mask\" user)",
        "    (39 \"F.Mask\" user)",
        "    (44 \"Edge.Cuts\" user)",
        "  )",
    ]

    # Net declarations
    for n, name in NETS.items():
        lines.append(f'  (net {n} "{name}")')

    lines.append("")

    # ── Components ───────────────────────────────────────────────────
    # All start on F.Cu.  The optimizer will re-assign layers.
    # Note: decoupling caps C1–C4 are initially clustered around U1 but
    # are movable — the optimizer should find B.Cu placement under U1.

    lines += u1_mcu()                                          # MCU — centre

    lines += u2_ldo()                                          # LDO — upper right

    lines += j1_usb()                                          # USB-C — left edge (fixed)
    lines += j2_swd()                                          # SWD header — right edge (fixed)

    lines += y1_crystal()                                      # Crystal — upper left

    # 100nF MCU decoupling caps — all near U1 initially
    lines += cap_0402("C1", "100nF", 3, 1, 13.0,  9.0)        # near U1 left-top pin
    lines += cap_0402("C2", "100nF", 3, 1, 17.0,  9.0)        # near U1 left-bottom
    lines += cap_0402("C3", "100nF", 3, 1, 13.0, 13.0)        # near U1 right-top
    lines += cap_0402("C4", "100nF", 3, 1, 17.0, 13.0)        # near U1 right-bottom

    # 10uF bulk caps for LDO
    lines += cap_0805("C5", "10uF",  17, 1, 24.0,  5.0)       # LDO input cap (VBUS_F)
    lines += cap_0805("C6", "10uF",   3, 1, 24.0, 16.0)       # LDO output cap (VCC)

    # Crystal load caps
    lines += cap_0402("C7", "18pF",   6, 1,  9.0,  9.0)       # XTAL_IN load cap
    lines += cap_0402("C8", "18pF",   7, 1, 13.0,  5.0)       # XTAL_OUT load cap

    # Resistors
    lines += res_0402("R1", "10k",    3, 12, 20.0, 18.0)       # nRESET pull-up
    lines += res_0402("R2", "33R",    4, 13,  7.0,  9.0)       # USB DP series
    lines += res_0402("R3", "33R",    5, 14,  7.0, 13.0)       # USB DM series
    lines += res_0402("R4", "100R",   8, 15, 11.0, 19.0)       # LED1 limiter
    lines += res_0402("R5", "100R",   9, 16, 19.0, 19.0)       # LED2 limiter

    # Status LEDs
    lines += led_0805("D1", "LED_RED",   15, 1, 9.0,  20.0)    # LED1
    lines += led_0805("D2", "LED_GREEN", 16, 1, 21.0, 20.0)    # LED2

    # Ferrite bead on VBUS
    lines += fb_0402("FB1", 5.0, 11.0)

    lines.append("")

    # ── Board outline: 30 × 22 mm ────────────────────────────────────
    lines += [
        '  (gr_line (start 0 0)   (end 30 0)   (layer "Edge.Cuts") (width 0.05))',
        '  (gr_line (start 30 0)  (end 30 22)  (layer "Edge.Cuts") (width 0.05))',
        '  (gr_line (start 30 22) (end 0 22)   (layer "Edge.Cuts") (width 0.05))',
        '  (gr_line (start 0 22)  (end 0 0)    (layer "Edge.Cuts") (width 0.05))',
        ")",
    ]

    return "\n".join(lines)


if __name__ == "__main__":
    import os
    out = os.path.join(os.path.dirname(__file__), "demo_two_layer.kicad_pcb")
    content = build_pcb()
    with open(out, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"[OK] Generated {out}")
    print(f"     {content.count(chr(10))+1} lines  |  {len(content)} bytes")
    print()
    print("Run the autoplacer with:")
    print("  python main.py tests/demo_two_layer.kicad_pcb --export tests/demo_two_layer_placed.kicad_pcb")
