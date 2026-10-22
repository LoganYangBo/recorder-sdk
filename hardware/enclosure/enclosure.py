"""Recorder enclosure: a thin slab that sticks to the back of a phone (MagSafe).

Run `python enclosure.py` (needs build123d) to write base.stl and lid.stl next to this file.
Every size is a parameter below; change them and run it again.

Looks: a soft-edged card, like the Plaud Note. The lid is a flush panel framed by the base's
edge, and everything you touch or look at sits on one line across the top:

    +-----------------------------------------------+
    |  [ screen window ]    ( )    :::::::::  USB-C >
    |                    button    mic / light grille|
    |                                            Bo   |
    +-----------------------------------------------+

Inside, everything lies side by side in one layer, each part in its own pocket, so nothing
needs glue:

    | OLED (glass up)      | button | XIAO     USB-C>
    | battery                    | SD module       |

The pockets' walls are lower than the parts, so the wires run over them. The lid panel drops
into the frame, rests on the inner ledge, presses the parts down and clicks in (bumps on its
long edges catch grooves in the frame). The button is a round pad cut free from the lid except
for a thin hinge; its nub presses the switch. Four thin disc magnets sit in the floor on the
MagSafe ring.
"""

from math import cos, radians, sin, sqrt
from pathlib import Path

from build123d import (
    Align,
    Axis,
    Box,
    BuildPart,
    BuildSketch,
    Circle,
    Compound,
    Cylinder,
    Locations,
    Mode,
    Plane,
    Pos,
    Rot,
    RectangleRounded,
    SlotOverall,
    Text,
    export_stl,
    extrude,
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
OLED_GLASS = (26.4, 12.2)  # the glass on the module's board (measured)
OLED_GLASS_FROM_END = 1.0  # glass edge from the board's end away from the pins (-x end)
BUTTON = (6.2, 6.2, 5.0)  # 6 x 6 tact switch: body, height to the top of the plunger
MAGNET = (8.0, 1.0)  # thin disc magnets: diameter, thickness
USB_C = (9.6, 3.6)  # the opening for the plug
USB_C_Z = 2.6  # its middle, above the pocket floor

# ---- case ----
WALL = 1.6  # side walls: the outer frame...
FRAME = 0.8  # ...of which this much rises around the lid; the rest is the lid's ledge
FLOOR_SKIN = 0.5  # plastic between the magnets and the phone
LID = 1.0
FIT = 0.2  # around each part
LID_FIT = 0.12  # around the lid panel
DIVIDER = 1.0  # pocket walls: thickness...
DIVIDER_H = 3.0  # ...and height (the wires run over them)
SNAP = 0.25  # how far the lid's snap bumps stick out
CORNER = 3.5  # corner radius, seen from above
TOP_EDGE = 0.6  # rounding of the frame's top edge
BOTTOM_EDGE = 1.5  # rounding of the edge against the phone (makes the slab look thinner)

WINDOW = (23.0, 6.5)  # the screen window: the lit area is about 22.4 x 5.6...
BEZEL = 1.2  # ...and the shallow step around it
# the button: "cap" is a separate key (print it in another colour) poking through the lid,
# "pad" is a round pad cut free from the lid itself
BUTTON_STYLE = "cap"
CAP = (6.6, 6.6)  # the key seen from above: equal for round, longer y for a pill
CAP_PROUD = 0.5  # how far it stands above the lid
CAP_FLANGE = (6.0, 12.0, 0.6)  # the bar under the lid that keeps it from falling out
BUTTON_PAD = 9.0  # "pad": its diameter...
SLIT = 0.6  # ...the cut around it...
HINGE = (3.0, 0.5)  # ...and where it stays joined to the lid: width, thinned to this thickness
GRILLE = (19.0, 13.0)  # the field of holes over the XIAO (mic, LEDs)
GRILLE_HOLE, GRILLE_PITCH = 1.0, 1.8
OWNER = ""  # engraved in the corner, e.g. "Bo"; "" for none

MAGSAFE_RADIUS = 25.0  # the phone's magnet ring, to the middle of the magnets
MAGNET_ANGLES = (35, 145, 215, 325)  # kept inside the width

# ---- layout ----
FLOOR = FLOOR_SKIN + MAGNET[1]
CAVITY_H = PART_H + 0.2


def pocket(size):
    return (size[0] + 2 * FIT, size[1] + 2 * FIT)


OLED_P, BTN_P, XIAO_P = pocket(OLED), pocket(BUTTON), pocket(XIAO)
BAT_P, SD_P = pocket(BATTERY), pocket(SD)
ROW1_W = max(OLED_P[1], BTN_P[1], XIAO_P[1])  # the top row, along +y
ROW2_W = max(BAT_P[1], SD_P[1])
INNER_L = max(OLED_P[0] + DIVIDER + BTN_P[0] + DIVIDER + XIAO_P[0], BAT_P[0] + DIVIDER + SD_P[0])
INNER_W = ROW1_W + DIVIDER + ROW2_W
OUTER = (INNER_L + 2 * WALL, INNER_W + 2 * WALL)
PANEL = (OUTER[0] - 2 * FRAME, OUTER[1] - 2 * FRAME)
THICKNESS = FLOOR + CAVITY_H + LID
LEDGE_Z = FLOOR + CAVITY_H  # the lid panel rests here
BUTTON_GAP = CAP_FLANGE[2] + 0.1 if BUTTON_STYLE == "cap" else 0.3  # above the plunger

x0, y1, y2 = -INNER_L / 2, INNER_W / 2 - ROW1_W / 2, -INNER_W / 2 + ROW2_W / 2
OLED_X = x0 + OLED_P[0] / 2
XIAO_X = INNER_L / 2 - XIAO_P[0] / 2  # against the +x wall, USB-C through it
BTN_X = (OLED_X + OLED_P[0] / 2 + XIAO_X - XIAO_P[0] / 2) / 2
BAT_X = x0 + BAT_P[0] / 2
SD_X = INNER_L / 2 - SD_P[0] / 2
GLASS_X = OLED_X - OLED[0] / 2 + OLED_GLASS_FROM_END + OLED_GLASS[0] / 2


def base():
    with BuildPart() as p:
        Box(*OUTER, THICKNESS, align=BOTTOM)
        fillet(p.edges().filter_by(Axis.Z), CORNER)
        fillet(p.edges().group_by(Axis.Z)[0], BOTTOM_EDGE)
        fillet(p.edges().group_by(Axis.Z)[-1], TOP_EDGE)
        # the recess the lid panel drops into, and the cavity under it
        with BuildSketch(Plane.XY.offset(LEDGE_Z)):
            RectangleRounded(*PANEL, CORNER - FRAME)
        extrude(amount=LID + 1, mode=Mode.SUBTRACT)
        with Locations(Pos(0, 0, FLOOR)):
            Box(INNER_L, INNER_W, CAVITY_H + 0.01, align=BOTTOM, mode=Mode.SUBTRACT)
        # magnets, flush with the floor (the parts lie on them)
        for a in MAGNET_ANGLES:
            x, y = MAGSAFE_RADIUS * cos(radians(a)), MAGSAFE_RADIUS * sin(radians(a))
            with Locations(Pos(x, y, FLOOR_SKIN)):
                Cylinder(MAGNET[0] / 2 + 0.1, MAGNET[1] + 0.1, align=BOTTOM, mode=Mode.SUBTRACT)
        # pocket walls: one between the rows, and the ones across each row
        with Locations(Pos(0, INNER_W / 2 - ROW1_W - DIVIDER / 2, FLOOR)):
            Box(INNER_L, DIVIDER, DIVIDER_H, align=BOTTOM)
        for x, y, w in (
            (OLED_X + OLED_P[0] / 2 + DIVIDER / 2, y1, ROW1_W),
            (XIAO_X - XIAO_P[0] / 2 - DIVIDER / 2, y1, ROW1_W),
            (BAT_X + BAT_P[0] / 2 + DIVIDER / 2, y2, ROW2_W),
        ):
            with Locations(Pos(x, y, FLOOR)):
                Box(DIVIDER, w, DIVIDER_H, align=BOTTOM)
        # the switch sits lower, so its plunger ends just under the lid (or the key's bar)
        sink = max(0.0, BUTTON[2] - (CAVITY_H - BUTTON_GAP))
        if sink:
            with Locations(Pos(BTN_X, y1, FLOOR - sink)):
                Box(*BTN_P, sink + 0.01, align=BOTTOM, mode=Mode.SUBTRACT)
        # USB-C through the +x wall: a rounded slot
        usb = Plane(
            origin=(OUTER[0] / 2 - WALL * 2, y1, FLOOR + USB_C_Z), x_dir=(0, 1, 0), z_dir=(1, 0, 0)
        )
        with BuildSketch(usb):
            SlotOverall(USB_C[0], USB_C[1])
        extrude(amount=WALL * 3, mode=Mode.SUBTRACT)
        # grooves for the lid's snap bumps, in the frame along the long sides
        for sx in (-1, 1):
            for side in (-1, 1):
                with Locations(Pos(sx * INNER_L / 4, side * PANEL[1] / 2, LEDGE_Z + LID / 2)):
                    Box(14, 2 * (SNAP + 0.1), 0.5, mode=Mode.SUBTRACT)
        # a slot at the -x end to get a fingernail under the lid
        with Locations(Pos(-PANEL[0] / 2, 0, LEDGE_Z)):
            Box(1.6, 8, 2 * LID, mode=Mode.SUBTRACT)
    return p.part


def pad():
    """The "pad" button: a round pad cut free from the lid except for a hinge."""
    r = BUTTON_PAD / 2
    with Locations(Pos(BTN_X, y1, -1)):
        Cylinder(r + SLIT, LID + 2, align=BOTTOM, mode=Mode.SUBTRACT)
    with Locations(Pos(BTN_X, y1, 0)):
        Cylinder(r, LID, align=BOTTOM)
    with Locations(Pos(BTN_X, y1 - r - SLIT / 2, LID / 2)):
        Box(HINGE[0], SLIT + 0.4, LID)
    with Locations(Pos(BTN_X, y1 - r - SLIT / 2, 0)):  # thinned from below so it bends
        Box(HINGE[0], SLIT + 3.0, LID - HINGE[1], align=BOTTOM, mode=Mode.SUBTRACT)
    with Locations(Pos(BTN_X, y1, LID)):  # a slight dish to find it by touch
        Cylinder(r - 1.2, 0.25, mode=Mode.SUBTRACT)
    with Locations(Pos(BTN_X, y1, -0.3)):
        Cylinder(1.5, 0.3, align=BOTTOM)  # the nub


def cap_outline(grow=0.0):
    w, h = CAP[0] + 2 * grow, CAP[1] + 2 * grow
    if h > w:
        SlotOverall(h, w, rotation=90)
    else:
        Circle(w / 2)


def cap():
    """The "cap" button: a key in the lid's hole, kept in by a bar under the lid."""
    with BuildPart() as p:
        with Locations(Pos(BTN_X, y1, LEDGE_Z - CAP_FLANGE[2])):
            Box(*CAP_FLANGE, align=BOTTOM)
        with BuildSketch(Plane.XY.offset(LEDGE_Z).move(Pos(BTN_X, y1))):
            cap_outline()
        extrude(amount=LID + CAP_PROUD)
        fillet(p.edges().group_by(Axis.Z)[-1], 0.4)
    return p.part


def lid():
    w, h = PANEL[0] - 2 * LID_FIT, PANEL[1] - 2 * LID_FIT
    with BuildPart() as p:
        with BuildSketch():
            RectangleRounded(w, h, CORNER - FRAME - LID_FIT)
        extrude(amount=LID)
        fillet(p.edges().group_by(Axis.Z)[-1], 0.3)
        for sx in (-1, 1):  # snap bumps
            for side in (-1, 1):
                with Locations(Pos(sx * INNER_L / 4, side * h / 2, LID / 2)):
                    Box(12, 2 * SNAP, 0.4)
        # the screen window, with a shallow step around it
        with BuildSketch(Plane.XY.offset(LID - 0.3).move(Pos(GLASS_X, y1))):
            RectangleRounded(WINDOW[0] + 2 * BEZEL, WINDOW[1] + 2 * BEZEL, 2.0)
        extrude(amount=1, mode=Mode.SUBTRACT)
        with BuildSketch(Plane.XY.offset(-1).move(Pos(GLASS_X, y1))):
            RectangleRounded(*WINDOW, 1.2)
        extrude(amount=LID + 2, mode=Mode.SUBTRACT)
        if BUTTON_STYLE == "cap":
            with BuildSketch(Plane.XY.offset(-1).move(Pos(BTN_X, y1))):
                cap_outline(0.2)
            extrude(amount=LID + 2, mode=Mode.SUBTRACT)
        else:
            pad()
        # the grille over the XIAO: a staggered field of holes, for its mic and LEDs
        gx, gy = GRILLE[0] / 2 - GRILLE_HOLE, GRILLE[1] / 2 - GRILLE_HOLE
        row_step = GRILLE_PITCH * sqrt(3) / 2
        holes = []
        for j in range(-int(gy / row_step), int(gy / row_step) + 1):
            shift = GRILLE_PITCH / 2 if j % 2 else 0.0
            for i in range(-int(gx / GRILLE_PITCH) - 1, int(gx / GRILLE_PITCH) + 2):
                x, y = i * GRILLE_PITCH + shift, j * row_step
                # inside a rounded rectangle, so the field has soft corners
                cx, cy = max(abs(x) - (gx - 3), 0), max(abs(y) - (gy - 3), 0)
                if abs(x) <= gx and cx * cx + cy * cy <= 9:
                    holes.append(Pos(XIAO_X + x, y1 + y, -1))
        with Locations(*holes):
            Cylinder(GRILLE_HOLE / 2, LID + 2, align=BOTTOM, mode=Mode.SUBTRACT)
        if OWNER:
            with BuildSketch(Plane.XY.offset(LID).move(Pos(PANEL[0] / 2 - 7, -PANEL[1] / 2 + 5))):
                Text(OWNER, 4.0)
            extrude(amount=-0.3, mode=Mode.SUBTRACT)
    return p.part


SETS = 2  # how many of each part go on the print plates
PLATE_GAP = 6.0


def lay_flat(part):
    """Centred on the origin, sitting on z = 0."""
    b = part.bounding_box()
    return Pos(-b.center().X, -b.center().Y, -b.min.Z) * part


def plate(parts):
    """The parts in rows (one row per kind), spaced out for one print."""
    placed, y = [], 0.0
    for part in parts:
        size = part.bounding_box().size
        for i in range(SETS):
            placed.append(
                Pos((i - (SETS - 1) / 2) * (size.X + PLATE_GAP), y - size.Y / 2, 0) * part
            )
        y -= size.Y + PLATE_GAP
    whole = Compound(placed)
    b = whole.bounding_box()
    return Pos(-b.center().X, -b.center().Y, 0) * whole


if __name__ == "__main__":
    # each part lying the way it prints: the lid face down, the button key on its bar
    parts = {"base": lay_flat(base()), "lid": lay_flat(Rot(180, 0, 0) * lid())}
    if BUTTON_STYLE == "cap":
        parts["button"] = lay_flat(cap())
    plates = {
        "plate-all": list(parts.values()),  # everything, one colour
        "plate-dark": [p for n, p in parts.items() if n != "lid"],  # base and button...
        "plate-light": [parts["lid"]],  # ...and lid in another colour
    }
    files = {**parts, **{name: plate(ps) for name, ps in plates.items()}}
    for name, part in files.items():
        export_stl(part, str(HERE / f"{name}.stl"), tolerance=0.02, angular_tolerance=0.2)
    print(f"outer {OUTER[0]:.1f} x {OUTER[1]:.1f} x {THICKNESS:.1f} mm")
