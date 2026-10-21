"""Recorder enclosure: a thin slab that sticks to the back of a phone (MagSafe).

Run `python enclosure.py` (needs build123d) to write base.stl and lid.stl next to this file.
Every size is a parameter below; change them and run it again.

Everything lies side by side in one layer, each part in its own pocket, so nothing needs glue:

    +----------------------------------------------+
    | button | battery              | XIAO     USB-C>
    |--------+----------------------+--------------|
    | OLED (glass up)          | SD module          |
    +----------------------------------------------+

The pockets' walls are lower than the parts, so the wires run over them. The lid snaps on
(bumps on its lip click into grooves in the base) and presses everything down. The button is a
tongue cut into the lid: press it and its nub presses the switch. Four thin disc magnets sit in
the floor on the MagSafe ring.
"""

from math import cos, radians, sin
from pathlib import Path

from build123d import (
    Align,
    Axis,
    Box,
    BuildPart,
    Cylinder,
    Locations,
    Mode,
    Pos,
    export_stl,
    fillet,
)

HERE = Path(__file__).parent
BOTTOM = (Align.CENTER, Align.CENTER, Align.MIN)

# ---- parts, measured (mm): length x width ----
PART_H = 4.6  # the thickest part (the XIAO with its USB-C)
BATTERY = (40.6, 19.8)
XIAO = (22.4, 17.8)
SD = (27.6, 17.7)  # with the card in
OLED = (38.3, 12.2)
OLED_GLASS = (30.0, 11.5)  # the glass on the module's board, shown through the window
OLED_GLASS_FROM_END = 1.0  # glass edge from the board's end away from the pins (-x end)
BUTTON = (6.2, 6.2, 5.0)  # 6 x 6 tact switch: body, height to the top of the plunger
MAGNET = (8.0, 1.0)  # thin disc magnets: diameter, thickness
USB_C = (9.6, 3.6)  # the opening for the plug
USB_C_Z = 2.6  # its middle, above the pocket floor

# ---- case ----
WALL = 1.4
FLOOR_SKIN = 0.5  # plastic between the magnets and the phone
LID = 1.0
FIT = 0.2  # around each part
DIVIDER = 1.0  # pocket walls: thickness...
DIVIDER_H = 3.0  # ...and height (the wires run over them)
LIP = 1.2  # the lid's lip, down inside the walls
SNAP = 0.35  # how far the snap bumps stick out
CORNER = 3.0

MAGSAFE_RADIUS = 25.0  # the phone's magnet ring, to the middle of the magnets
MAGNET_ANGLES = (35, 145, 215, 325)  # kept inside the 42 mm width

# ---- layout ----
FLOOR = FLOOR_SKIN + MAGNET[1]
CAVITY_H = PART_H + 0.2


def pocket(size):
    return (size[0] + 2 * FIT, size[1] + 2 * FIT)


BTN_P, BAT_P, XIAO_P = pocket(BUTTON), pocket(BATTERY), pocket(XIAO)
OLED_P, SD_P = pocket(OLED), pocket(SD)
ROW1_W = max(BTN_P[1], BAT_P[1], XIAO_P[1])
ROW2_W = max(OLED_P[1], SD_P[1])
INNER_L = max(BTN_P[0] + DIVIDER + BAT_P[0] + DIVIDER + XIAO_P[0], OLED_P[0] + DIVIDER + SD_P[0])
INNER_W = ROW1_W + DIVIDER + ROW2_W
OUTER = (INNER_L + 2 * WALL, INNER_W + 2 * WALL)
BASE_H = FLOOR + CAVITY_H
THICKNESS = BASE_H + LID

x0, y1, y2 = -INNER_L / 2, INNER_W / 2 - ROW1_W / 2, -INNER_W / 2 + ROW2_W / 2
BTN_X = x0 + BTN_P[0] / 2
BAT_X = x0 + BTN_P[0] + DIVIDER + BAT_P[0] / 2
XIAO_X = INNER_L / 2 - XIAO_P[0] / 2  # against the +x wall, USB-C through it
OLED_X = x0 + OLED_P[0] / 2
SD_X = INNER_L / 2 - SD_P[0] / 2
GLASS_X = OLED_X - OLED[0] / 2 + OLED_GLASS_FROM_END + OLED_GLASS[0] / 2


def base():
    with BuildPart() as p:
        Box(*OUTER, BASE_H, align=BOTTOM)
        fillet(p.edges().filter_by(Axis.Z), CORNER)
        with Locations(Pos(0, 0, FLOOR)):
            Box(INNER_L, INNER_W, CAVITY_H + 1, align=BOTTOM, mode=Mode.SUBTRACT)
        # magnets, flush with the floor (the parts lie on them)
        for a in MAGNET_ANGLES:
            x, y = MAGSAFE_RADIUS * cos(radians(a)), MAGSAFE_RADIUS * sin(radians(a))
            with Locations(Pos(x, y, FLOOR_SKIN)):
                Cylinder(MAGNET[0] / 2 + 0.1, MAGNET[1] + 0.1, align=BOTTOM, mode=Mode.SUBTRACT)
        # pocket walls: one between the rows, and the ones across each row
        with Locations(Pos(0, INNER_W / 2 - ROW1_W - DIVIDER / 2, FLOOR)):
            Box(INNER_L, DIVIDER, DIVIDER_H, align=BOTTOM)
        for x, y, w in (
            (x0 + BTN_P[0] + DIVIDER / 2, y1, ROW1_W),
            (XIAO_X - XIAO_P[0] / 2 - DIVIDER / 2, y1, ROW1_W),
            (OLED_X + OLED_P[0] / 2 + DIVIDER / 2, y2, ROW2_W),
            (SD_X - SD_P[0] / 2 - DIVIDER / 2, y2, ROW2_W),
        ):
            with Locations(Pos(x, y, FLOOR)):
                Box(DIVIDER, w, DIVIDER_H, align=BOTTOM)
        # the button sits lower, so its plunger ends just under the lid's tongue
        sink = max(0.0, BUTTON[2] - (CAVITY_H - 0.3))
        if sink:
            with Locations(Pos(BTN_X, y1, FLOOR - sink)):
                Box(*BTN_P, sink + 0.01, align=BOTTOM, mode=Mode.SUBTRACT)
        # USB-C through the +x wall
        with Locations(Pos(OUTER[0] / 2, y1, FLOOR + USB_C_Z)):
            Box(WALL * 3, USB_C[0], USB_C[1], mode=Mode.SUBTRACT)
        # grooves for the lid's snap bumps, on the long walls
        for sx in (-1, 1):
            for side in (-1, 1):
                with Locations(Pos(sx * INNER_L / 4, side * INNER_W / 2, BASE_H - LIP / 2)):
                    Box(14, 2 * (SNAP + 0.1), 0.7, mode=Mode.SUBTRACT)
        # a notch to get a fingernail under the lid
        with Locations(Pos(-OUTER[0] / 2, 0, BASE_H)):
            Box(WALL * 3, 8, 1.2, mode=Mode.SUBTRACT)
    return p.part


def lid():
    lip_l, lip_w = INNER_L - 2 * 0.15, INNER_W - 2 * 0.15
    with BuildPart() as p:
        Box(*OUTER, LID, align=BOTTOM)
        fillet(p.edges().filter_by(Axis.Z), CORNER)
        with Locations(Pos(0, 0, -LIP)):  # the lip, inside the walls
            Box(lip_l, lip_w, LIP, align=BOTTOM)
            Box(lip_l - 2.0, lip_w - 2.0, LIP, align=BOTTOM, mode=Mode.SUBTRACT)
        for sx in (-1, 1):  # snap bumps
            for side in (-1, 1):
                with Locations(Pos(sx * INNER_L / 4, side * (lip_w / 2), -LIP / 2)):
                    Box(12, 2 * SNAP, 0.5)
        # the OLED's window
        with Locations(Pos(GLASS_X, y2, 0)):
            Box(OLED_GLASS[0] - 1.0, OLED_GLASS[1] - 2.0, LID * 3, mode=Mode.SUBTRACT)
        # the button: a tongue (a U-shaped cut) with a nub underneath
        tongue = (10.0, 7.0)  # free end over the switch, hinged towards the middle
        tongue_x = BTN_X - 2.0 + tongue[0] / 2
        with Locations(Pos(tongue_x, y1, 0)):
            Box(tongue[0] + 1.2, tongue[1] + 1.2, LID * 3, mode=Mode.SUBTRACT)
            Box(tongue[0], tongue[1], LID, align=BOTTOM)
        with Locations(Pos(tongue_x + tongue[0] / 2 + 0.3, y1, 0)):  # its hinge
            Box(0.8, tongue[1], LID, align=BOTTOM)
        with Locations(Pos(BTN_X, y1, -0.3)):
            Cylinder(1.5, 0.3, align=BOTTOM)  # the nub
        # holes over the XIAO for its mic and LEDs (the recording light must show)
        for dx in range(-8, 9, 4):
            for dy in (-4, 0, 4):
                with Locations(Pos(XIAO_X + dx, y1 + dy, 0)):
                    Cylinder(0.6, LID * 3, mode=Mode.SUBTRACT)
    return p.part


if __name__ == "__main__":
    for name, part in (("base", base()), ("lid", lid())):
        export_stl(part, str(HERE / f"{name}.stl"))
    print(f"outer {OUTER[0]:.1f} x {OUTER[1]:.1f} x {THICKNESS:.1f} mm")
