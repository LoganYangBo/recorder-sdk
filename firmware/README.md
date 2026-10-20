# Recorder firmware

Arduino firmware for the wearable recorder: a Seeed XIAO nRF52840 Sense, a microSD module, a
button and a LiPo battery. It records all day to the card; `recorder sync` picks the files up.

## Hardware

| Part | XIAO pin |
|------|----------|
| SD module 3V3 / GND | 3V3 / GND |
| SD module CLK / MISO / MOSI / CS | D8 / D9 / D10 / D2 |
| Button (other side to GND) | D1 |
| Battery + (through the power switch) / − | BAT+ / BAT− pads on the back |

Use a 3.3 V SD module without a regulator, and a FAT32 card (32 GB or smaller formats as FAT32
on Windows). The battery only charges while the switch is on.

## Upload

1. Arduino IDE 2, board package **Seeed nRF52 mbed-enabled Boards** (boards manager URL
   `https://files.seeedstudio.com/arduino/package_seeeduino_boards_index.json`). Don't also
   install the non-mbed "Seeed nRF52 Boards": both have a board of the same name.
2. Board **XIAO nRF52840 Sense**, open `firmware/recorder/recorder.ino`, upload. The SD and
   PDM libraries come with the IDE and the board package.

## Use

Switching on starts recording. The button pauses and resumes.

| LED | Meaning |
|-----|---------|
| A blip every 5 s | Recording |
| Two blips every 5 s | Recording, but the clock isn't set |
| Dark | Paused (or switched off) |
| Solid for a moment after a press | Recording started |
| Two flashes after a press | Paused |
| Fast blinking | No card, or the card failed or is full. Press the button to retry |

Files are 16 kHz mono 16-bit WAV, a new one every 10 minutes (about 19 MB, 2.7 GB a day). The
header is rewritten every 30 seconds, so a power cut loses at most the last 30 seconds.

```
REC/20261020/093000.WAV     named by local start time (the SD library only does 8.3 names)
REC/NOTIME/00000001.WAV     recorded before the clock was set
```

## The clock

The board has no battery-backed clock. Plug it in with the switch on and run
`recorder settime` (or keep `recorder sync --watch` running, which sets it whenever the board
appears); this needs `pip install "recorder[device]"`. The clock then holds as long as the switch
stays on, so pause with the button rather than switching off. The serial protocol is described
in `src/recorder/clock.py`.

## Syncing

For now, take the card out and put it in a card reader. Create `recorder-device.json` with
`{"id": "recorder-1"}` at the card's root once (the firmware can't write that long name), then
`recorder sync` finds it like a plugged-in recorder.
