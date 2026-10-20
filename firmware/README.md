# Recorder firmware

Arduino firmware for the wearable recorder: a Seeed XIAO nRF52840 Sense, a microSD module, a
button and a LiPo battery. It listens all day and records to the card whenever there is sound;
`recorder sync --ble` fetches the files over Bluetooth (or take the card out and use
`recorder sync`), and `recorder live` transcribes what it hears as it happens.

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

1. Arduino IDE 2, boards manager URL
   `https://files.seeedstudio.com/arduino/package_seeeduino_boards_index.json`, board package
   **Seeed nRF52 Boards** (the one *without* "mbed-enabled": it has the Bluetooth library). If
   the mbed-enabled package is installed too, remove it, since both have a board of the same name.
2. Board **Seeed XIAO nRF52840 Sense**, open `firmware/recorder/recorder.ino`, upload. The SD,
   PDM and Bluetooth libraries come with the IDE and the board package. If the upload can't find
   the board, double-press its reset button and upload again.

## Use

Switching on starts listening. A recording starts when the sound rises well above the room's
background noise, keeps the second before it (so the first word isn't cut), and ends after
10 seconds of quiet: one file per stretch of sound, at most 10 minutes each. Any loud sound
triggers it, not only speech. The button pauses and resumes listening.

| LED | Meaning |
|-----|---------|
| A blip every 5 s | Listening |
| A blip every second | Recording |
| Blue instead of red | Connected over Bluetooth |
| Each blip doubled | The clock isn't set |
| Long blips | Battery below 15% |
| Dark | Paused (or switched off) |
| Solid for a moment after a press | Listening started |
| Two flashes after a press | Paused |
| Fast blinking | No card, or the card failed or is full. Press the button to retry |

Files are 16 kHz mono IMA ADPCM WAV, 4:1 compressed (about 29 MB per hour of sound). The header
is rewritten every 30 seconds, so a power cut loses at most the last 30 seconds.

```
REC/20261020/093000.WAV     named by local start time (the SD library only does 8.3 names)
REC/NOTIME/00000001.WAV     recorded before the clock was set
```

## Bluetooth

The recorder advertises as **Recorder**; no pairing is needed. On the computer
(`pip install "recorder[device]"`):

```
recorder device                          state, battery, recordings waiting
recorder sync --ble --watch --delete     fetch and transcribe every 10 minutes while in range
recorder live                            live transcription
```

`recorder ui` has a **Recorder** (设备) page that does all of this with buttons. Every
connection sets the clock. Downloads resume where they stopped; the file being recorded
comes in the next sync. The battery also shows in Windows' Bluetooth settings and in phone
apps such as nRF Connect (standard battery service). The protocol is described in
`src/recorder/ble.py`.

## Battery

The charge is estimated from the battery voltage, so it reads high while charging and jumps a
little under load. `recorder device` shows it, as does `?` in the serial monitor
(`bat=3.95 pct=69`).

## The clock

The board has no battery-backed clock. Bluetooth sets it on every connection; over USB, plug it
in with the switch on and run `recorder settime` (or keep `recorder sync --watch` running). The
clock then holds as long as the switch stays on, so pause with the button rather than switching
off.

## Tuning the trigger

Open the Arduino serial monitor (115200, New Line) and send `?`. The answer includes `level=`
(the sound right now) and `floor=` (the background noise it has learned). A frame counts as sound
when its level is over `SOUND_RATIO` (3) times the floor and over `MIN_LEVEL` (30); 5 such frames
within a quarter second start a recording. If it triggers on too little, raise those two at the
top of `recorder.ino`; if it misses quiet speech, lower them. `HANGOVER_BYTES` sets the
10 seconds of quiet that end a recording.

## Syncing with a card reader

Still works: create `recorder-device.json` with `{"id": "recorder-1"}` at the card's root once
(the firmware can't write that long name), then `recorder sync` finds it like a plugged-in
recorder.
