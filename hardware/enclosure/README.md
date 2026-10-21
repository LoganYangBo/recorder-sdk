# Enclosure

A box for the recorder that sticks to the back of a phone with MagSafe. Outside 56 × 32 × 19 mm.

![preview](preview.png)

`enclosure.py` builds it (`pip install build123d`, then `python enclosure.py`) and writes
`base.stl`, `lid.stl` and `button.stl`. Every size is a parameter at the top: measure the parts
with calipers, change the numbers, and run it again. This first version uses typical sizes.

## Inside, from the phone up

- **Floor**: four 6 × 2 mm disc magnets on the MagSafe ring, glued into pockets from inside,
  0.6 mm of plastic under them.
- **Battery** (602040), held by two low ribs.
- **XIAO** at the +x end, USB-C out through the end wall; the **SD module** at the other end.
  The mic hears through three small slots in each long wall, level with the board
  (`MIC_FROM_USB`: where the mic is along the board, measured from the USB end).
- **Under the lid**: the OLED behind its window, the button under its hole (with the printed
  cap), and a hole over the XIAO's LED so the recording light shows.

## Printing

- PETG (holds up in a warm car) or PLA, 0.2 mm layers, 3 walls, no supports.
- Base open side up; lid outside face down; the button cap flange down.
- First print the lid alone to check the screen window and button against the real parts.

## Assembly

1. Before gluing, hold each magnet against the phone to find the side that pulls, mark it, and
   glue it in with that side down (towards the phone). All four the same way.
2. Battery on the floor (a strip of Kapton or double-sided tape), boards on top of it.
3. OLED glass into the window from inside (a dot of glue on the board's corners), button glued
   under its hole with the cap in between.
4. Press the lid in. A strip of tape round the seam or a drop of glue makes it permanent.

Parts: 6 × 2 mm N52 disc magnets (a pack: some spare), super glue, Kapton tape. Without
MagSafe, stick a MagSafe ring sticker on the phone or its case.
