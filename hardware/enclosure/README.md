# Enclosure

A thin card for the recorder that sticks to the back of a phone with MagSafe:
73 × 43 × 7.3 mm.

![preview](preview.png)

`enclosure.py` builds it (`pip install build123d`, then `python enclosure.py`) and writes
`base.stl` and `lid.stl`. The part sizes at the top are measured; change any of them and run it
again.

## Look

Borrowed from card recorders like the Plaud Note and the Limitless pendant:

- **A soft-edged card.** Rounded corners, a rounded top edge, and a bigger rounding on the edge
  against the phone, so from the front it looks thinner than it is.
- **A flush lid panel** framed by the base's edge, so the seam reads as a design line. Print the
  base and lid in two colours (say graphite and sand) for the look in the preview.
- **One line across the top**: the screen window (with a shallow step around it), a round button,
  and a field of small holes over the XIAO for its mic and LEDs, which glow through it.
- **USB-C** is a rounded slot in the end wall.
- `OWNER` is engraved small in the corner (`""` for none).

## Inside

Everything lies side by side in one layer, each part in its own pocket, so nothing needs glue:

```
+-----------------------------------------------+
| OLED (glass up)      | button | XIAO     USB-C>
|-----------------------------------------------|
| battery                    | SD module        |
+-----------------------------------------------+
```

- **Pockets** fit each part with 0.2 mm to spare. Their walls are 3 mm high, lower than the
  parts, so the wires run over them.
- **The lid** drops into the frame, rests on the inner ledge and presses every part down. Bumps
  on its long edges click into grooves in the frame. A slot at the screen end lets a fingernail
  lift it.
- **The button** is a round pad cut free from the lid except for a thin hinge on the side toward
  the battery. Pressing it pushes a nub onto the switch, which sits a little lower in the floor.
- **Magnets**: four 8 × 1 mm discs in the floor, on the MagSafe ring, flush with the floor so the
  parts lie on them. 0.5 mm of plastic between them and the phone.

## Printing

- PETG or PLA, 0.2 mm layers (0.1 mm for the lid makes the hinge, holes and snaps nicer), no
  supports.
- Base flat on the bed; lid outside face down.
- Print the lid first and try it in the base: if the snaps are too tight or loose, change `SNAP`
  or `LID_FIT`.

## Assembly

1. Before putting each magnet in, hold it to the phone and mark the side that pulls. Put all
   four in with that side down. They are a press fit; a tiny drop of glue if one is loose.
2. Parts into their pockets, OLED glass up, the XIAO's USB-C against its opening.
3. Press the lid in.

Parts: 8 × 1 mm N52 disc magnets. Without MagSafe on the phone, a MagSafe ring sticker on the
phone or its case.
