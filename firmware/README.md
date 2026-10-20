# Recorder firmware

Arduino firmware for the wearable recorder: a Seeed XIAO nRF52840 Sense, a microSD module, a
button and a LiPo battery. It listens all day and records to the card whenever there is sound; `recorder sync` picks the files up.

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

Switching on starts listening. A recording starts when the sound rises well above the room's
background noise, keeps the second before it (so the first word isn't cut), and ends after
10 seconds of quiet: one file per stretch of sound, at most 10 minutes each. Any loud sound
triggers it, not only speech. The button pauses and resumes listening.

| LED | Meaning |
|-----|---------|
| A blip every 5 s | Listening |
| A blip every second | Recording |
| Each blip doubled | The clock isn't set |
| Dark | Paused (or switched off) |
| Solid for a moment after a press | Listening started |
| Two flashes after a press | Paused |
| Fast blinking | No card, or the card failed or is full. Press the button to retry |

Files are 16 kHz mono 16-bit WAV (115 MB per hour of sound). The header is rewritten every
30 seconds, so a power cut loses at most the last 30 seconds.

### Tuning the trigger

Open the Arduino serial monitor (115200, New Line) and send `?`. The answer ends with
`level=` (the sound right now) and `floor=` (the background noise it has learned). A frame counts
as sound when its level is over `SOUND_RATIO` (3) times the floor and over `MIN_LEVEL` (30);
5 such frames within a quarter second start a recording. If it triggers on too little, raise
those two at the top of `recorder.ino`; if it misses quiet speech, lower them. `HANGOVER_BYTES`
sets the 10 seconds of quiet that end a recording.

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
