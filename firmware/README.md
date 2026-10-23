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
| Battery + / − (a power switch in the + wire is optional) | BAT+ / BAT− pads on the back |
| Optional 0.91" OLED (128×32, SSD1306, I2C) VCC / GND / SDA / SCL | 3V3 / GND / D4 / D5 |

Use a 3.3 V SD module without a regulator, and a FAT32 card (32 GB or smaller formats as FAT32
on Windows). The recorder doesn't need a power switch: holding the button sleeps it (below).
With a switch, the battery only charges while it is on.

## Upload

1. Arduino IDE 2, boards manager URL
   `https://files.seeedstudio.com/arduino/package_seeeduino_boards_index.json`, board package
   **Seeed nRF52 Boards** (the one *without* "mbed-enabled": it has the Bluetooth library). If
   the mbed-enabled package is installed too, remove it, since both have a board of the same name.
2. Board **Seeed XIAO nRF52840 Sense**, open `firmware/recorder/recorder.ino`, upload. The card
   (SdFat), PDM and Bluetooth libraries come with the board package; the Arduino SD library
   isn't used any more. If the upload can't find the board, double-press its reset button and
   upload again.

The recorder runs a watchdog: if the firmware ever hangs, it restarts within 10 seconds (and
keeps its clock). Once started, the watchdog only stops at a reset or power cut, so after
uploading a firmware older than v4, press the reset button once.

Every push that touches `firmware/` is compiled by CI (`.github/workflows/firmware.yml`), and
`tests/test_firmware.py` runs the firmware itself on the computer (`firmware/sim`: the mic plays
generated sound, the card is a folder, and the SDK's Bluetooth code downloads from it).

## Use

Power-on starts listening. A recording starts when the sound rises well above the room's
background noise, keeps the second before it (so the first word isn't cut), and ends after
10 seconds of quiet: one file per stretch of sound, at most 10 minutes each. Any loud sound
triggers it, not only speech, but a file with under 0.64 s of sound in it (a knock, a door) is
deleted when it ends, and a Bluetooth sync drops files Whisper finds no speech in. The button:

- **Press**: pause or resume listening.
- **Hold for 2 seconds**: sleep (three slow blinks). Nothing is recorded, Bluetooth is off and
  the LED stays dark, but the clock keeps running and it charges when plugged in. Press to wake.

| LED | Meaning |
|-----|---------|
| A blip every 5 s | Listening |
| A blip every second | Recording |
| Blue instead of red | Connected over Bluetooth |
| Each blip doubled | The clock isn't set |
| Long blips | Battery below 15% |
| Dark | Paused, asleep or off |
| Three slow blinks | Going to sleep |
| Solid for a moment after a press | Listening started |
| Two flashes after a press | Paused |
| Fast blinking | No card, or the card failed or is full. It tries again every minute, or press the button |

With the OLED connected (and the **Adafruit SSD1306** library installed from the Library
Manager, with the libraries it asks for), the screen lights for 5 seconds at power-on and when
the button is pressed:

```
16:00            ᛒ  69% [■■■ ]     time, Bluetooth (blinks while sending), battery
● 0:05      ||||·||||||·||||       recording for 0:05 and the last few seconds of sound
Today 3 clips 12 min               recordings kept today
```

The middle line is `LISTEN` (with the sound wave: about half height starts a recording),
`PAUSED` or `NO CARD`. Power-on and waking play a short hello (sound bars and a greeting with
the name in `OWNER` at the top of the screen section); going to sleep shows a moon and
"Good night". With a screen the first press only lights it; pressing again while it's lit
pauses or resumes. Dark, it draws almost nothing. Without a screen the
firmware works as before (without the library, set `USE_SCREEN` to 0 at the top of
`recorder.ino`).

Files are 16 kHz mono IMA ADPCM WAV, 4:1 compressed (about 29 MB per hour of sound). The header
is rewritten every 30 seconds, so a power cut loses at most the last 30 seconds. The card gets
the audio in 8 KB multi-sector writes that end on sector boundaries, which keeps the card and
the CPU busy for a fraction of the time single-sector writes did.

```
REC/20261020/093000.WAV     named by local start time, with the date as the folder
REC/NOTIME/00000001.WAV     recorded before the clock was set
recorder-device.json        written at power-on if missing: marks the card for `recorder sync`
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
comes in the next sync, and deleting a day's last file removes its folder too.

Downloads keep up to 16 notifications queued in the radio (the Bluefruit default for
`BANDWIDTH_MAX` is 3, which caps each connection event at three packets), so the computer can
take as many per event as it likes. `?` reports `txq=16` (or 3, if the SoftDevice's RAM was
too small for the larger queue) and the connection interval in use (`ci=7.5`). The link asks
for 7.5 ms while sending and drops to 100-200 ms after 15 s without a transfer or live audio,
which costs the radio far less while `recorder ui` stays connected. The battery also shows in Windows' Bluetooth settings and in phone
apps such as nRF Connect (standard battery service). The protocol is described in
`src/recorder/ble.py`.

## Battery

The recorder charges over USB (any port or charger, same speed) at 100 mA: a 400 mAh cell
fills in about 4-5 hours. The green light is on while it charges.

Firmware v4 turns on the nRF52840's DC/DC converter (the XIAO has its inductor), which cuts
the chip's own current by about 40% whenever the CPU or radio runs, and leaves the unused
motion sensor unpowered.

The charge is estimated from the battery voltage, so it reads high while charging and jumps a
little under load. `recorder device` shows it, as does `?` in the serial monitor
(`bat=3.95 pct=69`).

## The clock

The board has no battery-backed clock. Bluetooth sets it on every connection; over USB, plug it
in and run `recorder settime` (or keep `recorder sync --watch` running). The clock then holds
while the recorder has power, asleep included, and through a reset (the button, or the
watchdog: `?` says which in `reset=`); only a flat battery, a power switch or a firmware upload
loses it.

## Tuning the trigger

Open the Arduino serial monitor (115200, New Line) and send `?`. The answer includes `level=`
(the sound right now) and `floor=` (the background noise it has learned). A frame counts as sound
when its level is over `SOUND_RATIO` (3) times the floor and over `MIN_LEVEL` (30); 5 such frames
within a quarter second start a recording. If it triggers on too little, raise those two at the
top of `recorder.ino`; if it misses quiet speech, lower them. `HANGOVER_BYTES` sets the
10 seconds of quiet that end a recording.

Firmware v4 sets the mic's gain to 0 dB (`PDM_GAIN`), 10 dB above the library's default, which
left quiet speech only a few steps above the ADPCM coder's smallest step: levels read about 3
times what they did, so `MIN_LEVEL` 30 now catches quieter voices than before.

## Syncing with a card reader

Still works: the firmware writes `recorder-device.json` (`{"id": "recorder-xxxx"}`, from its
Bluetooth address) at the card's root if it isn't there, so `recorder sync` finds the card like
a plugged-in recorder. A marker you made yourself is kept.
