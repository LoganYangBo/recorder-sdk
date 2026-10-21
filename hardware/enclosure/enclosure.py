"""Recorder enclosure: a box that sticks to the back of a phone (MagSafe).

Run `python enclosure.py` (needs build123d) to write base.stl, lid.stl and button.stl next to
this file. Every size is a parameter below: measure the parts with calipers and change them.

Stack, from the phone up:
    floor with four 6 x 2 mm disc magnets on the MagSafe ring (glued in from inside)
    battery (602040)
    XIAO (USB-C out of the +x end) and the SD module, side by side
    under the lid: the OLED behind its window (over the SD end), the button and the LED hole
    (over the XIAO); the mic hears through holes in both long walls, level with it
The lid presses into the base on a lip (a drop of glue or a strip of tape holds it for good).
"""

from math import cos, radians, sin
from pathlib import Path

from build123d import (
    Axis,
    Align,
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

# ---- parts (mm) -- measure yours ----
BATTERY = (40.0, 20.0, 6.3)  # 602040, a little swollen when full
XIAO = (21.0, 17.8, 4.2)  # board, height with the USB-C plug and parts
SD = (18.5, 14.5, 3.5)  # mini SD module with the card in
OLED = (38.0, 12.0, 3.2)  # 0.91" module: board, height with the glass
OLED_VIEW = (23.0, 6.5)  # the lit area, centred on the glass
OLED_VIEW_SHIFT = -2.5  # the glass sits towards one end of the board (+x)
BUTTON = (6.2, 6.2, 5.0)  # 6 x 6 tact switch, body + plunger height
MAGNET = (6.2, 2.1)  # disc magnet diameter, thickness (+ clearance)
USB_C = (9.6, 3.8)  # the opening for the plug

# ---- case ----
WALL = 1.6
FLOOR_SKIN = 0.6  # plastic between the magnets and the phone: thinner holds better
LID = 1.4
LIP = 1.6  # how deep the lid's lip goes into the base
FIT = 0.25  # clearance for parts and the lip
CORNER = 3.0

MAGSAFE_RADIUS = 25.0  # the phone's magnet ring, to the middle of the magnets
MAGNET_ANGLES = (25, -25, 155, -155)  # where on the ring the four magnets sit

INNER_L = 52.8
INNER_W = 28.8
FLOOR = FLOOR_SKIN + MAGNET[1]
BOARDS_Z = FLOOR + BATTERY[2] + 0.4  # where the XIAO and SD module sit
TOP_LAYER = max(OLED[2], BUTTON[2] - LID) + 0.6  # under the lid: OLED and button
INNER_H = BATTERY[2] + 0.4 + max(XIAO[2], SD[2]) + TOP_LAYER
OUTER = (INNER_L + 2 * WALL, INNER_W + 2 * WALL)
BASE_H = FLOOR + INNER_H

# positions in the cavity (x along the length, 0 = middle)
XIAO_X = INNER_L / 2 - XIAO[0] / 2 - 0.5  # against the USB end
SD_X = -INNER_L / 2 + SD[0] / 2 + 0.5
OLED_X = -INNER_L / 2 + OLED[0] / 2 + 1.0
BUTTON_X = INNER_L / 2 - 8.0
MIC_FROM_USB = 16.0  # where the mic is along the XIAO, from its USB end: measure it
MIC_X = INNER_L / 2 - 0.5 - MIC_FROM_USB
LED_HOLE = (INNER_L / 2 - 3.5, 6.0)  # over the XIAO's LED, by the USB end: it must show


def base():
    with BuildPart() as p:
        Box(*OUTER, BASE_H, align=(Align.CENTER, Align.CENTER, Align.MIN))
        fillet(p.edges().filter_by(Axis.Z), CORNER)
        with Locations(Pos(0, 0, FLOOR)):
            Box(
                INNER_L,
                INNER_W,
                INNER_H + 1,
                align=(Align.CENTER, Align.CENTER, Align.MIN),
                mode=Mode.SUBTRACT,
            )
        for a in MAGNET_ANGLES:
            x, y = MAGSAFE_RADIUS * cos(radians(a)), MAGSAFE_RADIUS * sin(radians(a))
            with Locations(Pos(x, y, FLOOR_SKIN)):
                Cylinder(
                    MAGNET[0] / 2,
                    MAGNET[1] + 0.1,
                    align=(Align.CENTER, Align.CENTER, Align.MIN),
                    mode=Mode.SUBTRACT,
                )
        # USB-C through the +x wall, level with the XIAO's plug
        with Locations(Pos(OUTER[0] / 2, 0, BOARDS_Z + 1.2 + USB_C[1] / 2)):
            Box(WALL * 3, USB_C[0], USB_C[1], mode=Mode.SUBTRACT)
        # mic: three small holes through each long wall, level with the board
        for dx in (-2, 0, 2):
            for side in (-1, 1):
                with Locations(Pos(MIC_X + dx, side * OUTER[1] / 2, BOARDS_Z + 2.0)):
                    Box(1.0, WALL * 3, 1.0, mode=Mode.SUBTRACT)
        # low ribs that keep the battery from sliding
        for sx in (-1, 1):
            with Locations(Pos(sx * (BATTERY[0] / 2 + FIT + 0.6), 0, FLOOR)):
                Box(1.2, BATTERY[1] * 0.6, 2.0, align=(Align.CENTER, Align.CENTER, Align.MIN))
    return p.part


def lid():
    lip_l, lip_w = INNER_L - 2 * FIT, INNER_W - 2 * FIT
    with BuildPart() as p:
        Box(*OUTER, LID, align=(Align.CENTER, Align.CENTER, Align.MIN))
        fillet(p.edges().filter_by(Axis.Z), CORNER)
        with Locations(Pos(0, 0, -LIP)):  # the lip goes down into the base
            Box(lip_l, lip_w, LIP, align=(Align.CENTER, Align.CENTER, Align.MIN))
            Box(
                lip_l - 2.4,
                lip_w - 2.4,
                LIP,
                align=(Align.CENTER, Align.CENTER, Align.MIN),
                mode=Mode.SUBTRACT,
            )
        # OLED window, and a pocket underneath that holds the module's board
        with Locations(Pos(OLED_X + OLED_VIEW_SHIFT, 0, 0)):
            Box(*OLED_VIEW, LID * 3, mode=Mode.SUBTRACT)
        with Locations(Pos(BUTTON_X, 0, 0)):
            Cylinder(2.2, LID * 3, mode=Mode.SUBTRACT)  # the cap's stem goes through
        with Locations(Pos(*LED_HOLE, 0)):
            Cylinder(1.0, LID * 3, mode=Mode.SUBTRACT)
    return p.part


def button_cap():
    """A cap that sits in the lid's hole and presses the switch underneath."""
    with BuildPart() as p:
        Cylinder(2.0, LID + 0.8, align=(Align.CENTER, Align.CENTER, Align.MIN))  # the stem
        Cylinder(3.0, 0.6, align=(Align.CENTER, Align.CENTER, Align.MIN))  # a flange inside
    return p.part


if __name__ == "__main__":
    for name, part in (("base", base()), ("lid", lid()), ("button", button_cap())):
        export_stl(part, str(HERE / f"{name}.stl"))
    print(f"outer {OUTER[0]:.1f} x {OUTER[1]:.1f} x {BASE_H + LID:.1f} mm")
