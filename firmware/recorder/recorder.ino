// Recorder firmware v2 for the Seeed XIAO nRF52840 Sense.
// Board package: "Seeed nRF52 mbed-enabled Boards", board "XIAO nRF52840 Sense".
//
// Listens to the onboard PDM mic all the time and records to the microSD card only while
// there is sound: a recording starts when the level rises well above the room's noise floor
// (keeping the second before it), and ends after 10 s of quiet. 16 kHz mono 16-bit WAV, at
// most 10 minutes per file. The button pauses and resumes listening. Wiring (see
// firmware/README.md): SD on SPI (D8 SCK, D9 MISO, D10 MOSI, CS on D2), button between D1 and GND.
//
// Files are named by local start time, REC/20261020/093000.WAV (the SD library only does
// 8.3 names, so the date is the folder). Without a clock they go to REC/NOTIME/00000001.WAV.
// The board has no battery-backed clock: the computer sets it over USB serial
// (`recorder settime`, or `recorder sync --watch` when the board is plugged in), and it holds
// while the power switch stays on.

#include <SPI.h>
#include <SD.h>
#include <PDM.h>
#include <time.h>

extern "C" void set_time(time_t t);  // mbed's RTC (lives on while powered)

const uint32_t SAMPLE_RATE = 16000;
const uint32_t BYTES_PER_SEC = SAMPLE_RATE * 2;
const uint32_t CHUNK_BYTES = BYTES_PER_SEC * 600;  // at most 10 minutes per file
const uint32_t FLUSH_BYTES = BYTES_PER_SEC * 30;   // header + flush every 30 s: a power cut loses at most this
const uint32_t WRITE_BLOCK = 4096;                 // bytes per SD write (128 ms of audio)
const int SD_CS = D2;
const int BUTTON = D1;
const time_t VALID_AFTER = 1700000000;  // anything earlier means the clock was never set

// Sound detection, on 16 ms frames. Tune with the level= and floor= that `?` reports.
const uint32_t FRAME = 512;                          // bytes: 256 samples
const uint32_t PREROLL_BYTES = BYTES_PER_SEC;        // keep 1 s before the sound that starts a recording
const uint32_t HANGOVER_BYTES = BYTES_PER_SEC * 10;  // end it after 10 s without sound
const float SOUND_RATIO = 3.0;  // a frame is sound when this many times above the noise floor...
const float MIN_LEVEL = 30;     // ...and at least this loud (mean distance from silence)
const int START_FRAMES = 5;     // sound frames among the last 16 that start a recording

// Ring buffer between the PDM interrupt and the SD writes: 4 s of audio holds the pre-roll
// plus SD stalls and opening files. Indices run freely; the size is a power of two. `scan` is
// how far sound detection has got; `tail` how far the SD writes have (or, while listening,
// the start of the pre-roll).
const uint32_t RING_SIZE = 131072;
alignas(4) uint8_t ring[RING_SIZE];
volatile uint32_t head = 0, tail = 0, dropped = 0;
uint32_t scan = 0, lastSoundAt = 0;
uint16_t recent = 0;  // one bit per recent frame, 1 = sound
float level = 0, noiseFloor = 0;
int16_t pdmBuf[256];

enum State { PAUSED, LISTENING, RECORDING, FAILED };
State state = PAUSED;
File wav;
uint32_t dataBytes = 0, sinceFlush = 0, notimeIndex = 0;
char path[32];
uint8_t block[WRITE_BLOCK];
uint32_t feedbackUntil = 0;
bool feedbackOn = false;

void onPDMdata() {
  int n = PDM.available();
  if (n > (int)sizeof(pdmBuf)) n = sizeof(pdmBuf);
  PDM.read(pdmBuf, n);
  if (RING_SIZE - (head - tail) < (uint32_t)n) {
    dropped += n;
    return;
  }
  uint32_t pos = head & (RING_SIZE - 1);
  uint32_t first = min((uint32_t)n, RING_SIZE - pos);
  memcpy(ring + pos, pdmBuf, first);
  memcpy(ring, (uint8_t*)pdmBuf + first, n - first);
  head += n;
}

void take(uint8_t* out, uint32_t n) {
  uint32_t pos = tail & (RING_SIZE - 1);
  uint32_t first = min(n, RING_SIZE - pos);
  memcpy(out, ring + pos, first);
  memcpy(out + first, ring, n - first);
  tail += n;
}

// Looks at the frame at `scan` (frames never wrap: scan steps by FRAME, which divides
// RING_SIZE) and tells whether it is sound. The noise floor follows quiet quickly and loud
// slowly, so a room that stays louder for a minute or so becomes the new quiet.
bool soundFrame() {
  const int16_t* x = (const int16_t*)(ring + (scan & (RING_SIZE - 1)));
  const int n = FRAME / 2;
  int32_t sum = 0;
  for (int i = 0; i < n; i++) sum += x[i];
  int32_t mean = sum / n;
  uint32_t dist = 0;
  for (int i = 0; i < n; i++) dist += abs(x[i] - mean);
  level = (float)dist / n;
  scan += FRAME;
  if (noiseFloor == 0) noiseFloor = level;
  bool sound = level > noiseFloor * SOUND_RATIO && level > MIN_LEVEL;
  float rate = level < noiseFloor ? 0.05f : sound ? 0.0002f : 0.001f;
  noiseFloor += (level - noiseFloor) * rate;
  return sound;
}

// ---- clock ----

bool clockValid() { return time(NULL) > VALID_AFTER; }

void stamp(char* out, size_t size) {
  if (!clockValid()) {
    snprintf(out, size, "unset");
    return;
  }
  time_t t = time(NULL);
  strftime(out, size, "%Y%m%dT%H%M%S", gmtime(&t));  // the clock holds local time
}

// ---- WAV files ----

void put32(File& f, uint32_t v) { f.write((uint8_t*)&v, 4); }
void put16(File& f, uint16_t v) { f.write((uint8_t*)&v, 2); }

void writeHeader(File& f, uint32_t bytes) {
  f.write((const uint8_t*)"RIFF", 4);
  put32(f, 36 + bytes);
  f.write((const uint8_t*)"WAVEfmt ", 8);
  put32(f, 16);
  put16(f, 1);  // PCM
  put16(f, 1);  // mono
  put32(f, SAMPLE_RATE);
  put32(f, BYTES_PER_SEC);
  put16(f, 2);
  put16(f, 16);
  f.write((const uint8_t*)"data", 4);
  put32(f, bytes);
}

// The file for audio that started `back` seconds ago.
void nextPath(uint32_t back) {
  if (clockValid()) {
    time_t t = time(NULL) - back;
    struct tm* g = gmtime(&t);
    char dir[16];
    snprintf(dir, sizeof(dir), "REC/%04d%02d%02d", g->tm_year + 1900, g->tm_mon + 1, g->tm_mday);
    SD.mkdir(dir);
    snprintf(path, sizeof(path), "%s/%02d%02d%02d.WAV", dir, g->tm_hour, g->tm_min, g->tm_sec);
    for (char c = 'A'; SD.exists(path) && c <= 'Z'; c++)  // same second again: 093000A.WAV
      snprintf(path, sizeof(path), "%s/%02d%02d%02d%c.WAV", dir, g->tm_hour, g->tm_min,
               g->tm_sec, c);
    return;
  }
  SD.mkdir("REC/NOTIME");
  do {
    snprintf(path, sizeof(path), "REC/NOTIME/%08lu.WAV", (unsigned long)++notimeIndex);
  } while (SD.exists(path));
}

bool startFile() {
  nextPath((head - tail) / BYTES_PER_SEC);  // what's still buffered started that long ago
  File f = SD.open(path, FILE_WRITE);
  if (!f) return false;
  writeHeader(f, 0);
  f.close();
  wav = SD.open(path, O_READ | O_WRITE);  // not FILE_WRITE: that appends, and we rewrite the header
  if (!wav) return false;
  wav.seek(44);
  dataBytes = sinceFlush = 0;
  return true;
}

void saveHeader() {
  wav.seek(0);
  writeHeader(wav, dataBytes);
  wav.seek(44 + dataBytes);
  wav.flush();
  sinceFlush = 0;
}

void finishFile() {
  if (!wav) return;
  saveHeader();
  wav.close();
}

bool writeBlock(uint32_t n) {
  take(block, n);
  if (wav.write(block, n) != n) return false;
  dataBytes += n;
  sinceFlush += n;
  if (sinceFlush >= FLUSH_BYTES) saveHeader();
  if (dataBytes >= CHUNK_BYTES) {
    finishFile();
    return startFile();
  }
  return true;
}

// Writes the buffered audio up to `end` and closes the file.
bool drainAndFinish(uint32_t end) {
  uint32_t left = end - tail;
  while (left > 0) {
    uint32_t n = min(left, WRITE_BLOCK);
    if (!writeBlock(n)) return false;
    left -= n;
  }
  finishFile();
  return true;
}

// ---- states ----

void fail() {
  PDM.end();
  if (wav) wav.close();
  state = FAILED;
}

void startListening() {
  head = tail = scan = 0;
  recent = 0;
  noiseFloor = 0;  // learn the room again
  if (!PDM.begin(1, SAMPLE_RATE)) return fail();
  state = LISTENING;
}

void pause() {
  PDM.end();
  if (state == RECORDING && !drainAndFinish(head)) return fail();
  state = PAUSED;
}

void startSegment() {
  if (!startFile()) return fail();
  lastSoundAt = scan;
  state = RECORDING;
}

void endSegment() {
  if (!drainAndFinish(scan)) return fail();  // the rest is the next pre-roll
  recent = 0;
  state = LISTENING;
}

void onPress() {
  if (state == LISTENING || state == RECORDING) {
    pause();
  } else {
    if (state == FAILED) {  // card back in? start over
      SD.end();
      if (!SD.begin(SD_CS)) return;
    }
    startListening();
  }
  feedbackOn = state != PAUSED;
  feedbackUntil = millis() + 600;
}

// ---- button, LED, serial ----

void handleButton() {
  static bool last = HIGH, stable = HIGH;
  static uint32_t changedAt = 0;
  bool now = digitalRead(BUTTON);
  if (now != last) {
    last = now;
    changedAt = millis();
  }
  if (millis() - changedAt > 30 && now != stable) {
    stable = now;
    if (stable == LOW) onPress();
  }
}

// LED (on = LOW): listening blips every 5 s, recording every second (each blip doubled if the
// clock isn't set), paused is dark, a failure (no card, card full) blinks fast.
// After a press: solid = listening, two flashes = paused.
void updateLed() {
  uint32_t ms = millis();
  bool on;
  if ((int32_t)(feedbackUntil - ms) > 0) {
    uint32_t p = 600 - (feedbackUntil - ms);
    on = feedbackOn || p < 100 || (p >= 200 && p < 300);
  } else if (state == FAILED) {
    on = (ms / 100) % 2;
  } else if (state == LISTENING || state == RECORDING) {
    uint32_t p = ms % (state == RECORDING ? 1000 : 5000);  // long blips: the loop may be busy writing
    on = p < 150 || (!clockValid() && p >= 400 && p < 550);
  } else {
    on = false;
  }
  digitalWrite(LED_BUILTIN, on ? LOW : HIGH);
}

// One line per command, see src/recorder/clock.py:
//   ?           -> RECORDER 2 <20261020T093000|unset> <LISTEN|REC|PAUSED|FAILED> <file|->
//                  dropped=<n> level=<now> floor=<noise floor>
//   T<seconds>  -> OK 20261020T093000   (local wall-clock time, counted as if it were UTC)
void command(const char* line) {
  char now[20];
  if (line[0] == 'T') {
    time_t t = (time_t)strtoull(line + 1, NULL, 10);
    if (t <= VALID_AFTER) {
      Serial.println("ERR bad time");
      return;
    }
    bool wasValid = clockValid();
    set_time(t);
    stamp(now, sizeof(now));
    Serial.print("OK ");  // answer first: switching files below takes a while
    Serial.println(now);
    if (state == RECORDING && !wasValid) {  // go on in a properly named file
      finishFile();
      if (!startFile()) fail();
    }
  } else if (line[0] == '?') {
    stamp(now, sizeof(now));
    const char* names[] = {"PAUSED", "LISTEN", "REC", "FAILED"};
    Serial.print("RECORDER 2 ");
    Serial.print(now);
    Serial.print(" ");
    Serial.print(names[state]);
    Serial.print(" ");
    Serial.print(state == RECORDING ? path : "-");
    Serial.print(" dropped=");
    Serial.print(dropped);
    Serial.print(" level=");
    Serial.print(level, 0);
    Serial.print(" floor=");
    Serial.println(noiseFloor, 0);
  }
}

void handleSerial() {
  static char line[40];
  static int len = 0;
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n' || c == '\r') {
      if (len) {
        line[len] = 0;
        command(line);
        len = 0;
      }
    } else if (len < (int)sizeof(line) - 1) {
      line[len++] = c;
    }
  }
}

void setup() {
  Serial.begin(115200);  // doesn't wait for the computer: on battery there is none
  pinMode(LED_BUILTIN, OUTPUT);
  digitalWrite(LED_BUILTIN, HIGH);
  pinMode(BUTTON, INPUT_PULLUP);
  PDM.onReceive(onPDMdata);
  if (!SD.begin(SD_CS)) {
    state = FAILED;
    return;
  }
  startListening();
}

void loop() {
  handleSerial();
  handleButton();
  updateLed();
  while ((state == LISTENING || state == RECORDING) && head - scan >= FRAME) {
    bool sound = soundFrame();
    recent = (recent << 1) | sound;
    if (state == LISTENING) {
      if (scan - tail > PREROLL_BYTES) tail = scan - PREROLL_BYTES;  // forget older audio
      if (__builtin_popcount(recent) >= START_FRAMES) startSegment();
    } else if (sound) {
      lastSoundAt = scan;
    } else if (scan - lastSoundAt > HANGOVER_BYTES) {
      endSegment();
    }
  }
  if (state == RECORDING && scan - tail >= WRITE_BLOCK) {  // writes never pass detection
    if (!writeBlock(WRITE_BLOCK)) fail();
  } else {
    delay(10);  // lets the CPU sleep between blocks
  }
}
