// Recorder firmware v1 for the Seeed XIAO nRF52840 Sense.
// Board package: "Seeed nRF52 mbed-enabled Boards", board "XIAO nRF52840 Sense".
//
// Records the onboard PDM mic continuously to the microSD card as 16 kHz mono 16-bit WAV,
// one file per 10 minutes. The button pauses and resumes. Wiring (see firmware/README.md):
// SD on SPI (D8 SCK, D9 MISO, D10 MOSI, CS on D2), button between D1 and GND.
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
const uint32_t CHUNK_BYTES = BYTES_PER_SEC * 600;  // a new file every 10 minutes
const uint32_t FLUSH_BYTES = BYTES_PER_SEC * 30;   // header + flush every 30 s: a power cut loses at most this
const uint32_t WRITE_BLOCK = 4096;                 // bytes per SD write (128 ms of audio)
const int SD_CS = D2;
const int BUTTON = D1;
const time_t VALID_AFTER = 1700000000;  // anything earlier means the clock was never set

// Ring buffer between the PDM interrupt and the SD writes: 2 s of audio covers SD stalls and
// closing/opening files. Indices run freely; the size is a power of two.
const uint32_t RING_SIZE = 65536;
uint8_t ring[RING_SIZE];
volatile uint32_t head = 0, tail = 0, dropped = 0;
int16_t pdmBuf[256];

enum State { PAUSED, RECORDING, FAILED };
State state = PAUSED;
File wav;
uint32_t dataBytes = 0, sinceFlush = 0, notimeIndex = 0;
char path[32];
uint8_t block[WRITE_BLOCK];
uint32_t feedbackUntil = 0;
bool feedbackStart = false;

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

void nextPath() {
  if (clockValid()) {
    time_t t = time(NULL);
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
  nextPath();
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

// ---- states ----

void fail() {
  PDM.end();
  if (wav) wav.close();
  state = FAILED;
}

void startRecording() {
  if (!startFile()) return fail();
  head = tail = 0;
  if (!PDM.begin(1, SAMPLE_RATE)) return fail();
  state = RECORDING;
}

void stopRecording() {
  PDM.end();
  uint32_t left = head - tail;
  while (left > 0) {  // keep what's still in the buffer
    uint32_t n = min(left, WRITE_BLOCK);
    if (!writeBlock(n)) return fail();
    left -= n;
  }
  finishFile();
  state = PAUSED;
}

void onPress() {
  if (state == RECORDING) {
    stopRecording();
  } else {
    if (state == FAILED) {  // card back in? start over
      SD.end();
      if (!SD.begin(SD_CS)) return;
    }
    startRecording();
  }
  feedbackStart = state == RECORDING;
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

// LED (on = LOW): recording blips every 5 s (twice if the clock isn't set), paused is dark,
// a failure (no card, card full) blinks fast. After a press: solid = recording, two flashes = paused.
void updateLed() {
  uint32_t ms = millis();
  bool on;
  if ((int32_t)(feedbackUntil - ms) > 0) {
    uint32_t p = 600 - (feedbackUntil - ms);
    on = feedbackStart || p < 100 || (p >= 200 && p < 300);
  } else if (state == FAILED) {
    on = (ms / 100) % 2;
  } else if (state == RECORDING) {
    uint32_t p = ms % 5000;  // blips long enough to survive a loop busy with an SD write
    on = p < 150 || (!clockValid() && p >= 400 && p < 550);
  } else {
    on = false;
  }
  digitalWrite(LED_BUILTIN, on ? LOW : HIGH);
}

// One line per command, see src/recorder/clock.py:
//   ?           -> RECORDER 1 <20261020T093000|unset> <REC|PAUSED|FAILED> <file> dropped=<n>
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
    if (state == RECORDING && !wasValid) {  // start a properly named file right away
      finishFile();
      if (!startFile()) fail();
    }
    stamp(now, sizeof(now));
    Serial.print("OK ");
    Serial.println(now);
  } else if (line[0] == '?') {
    stamp(now, sizeof(now));
    const char* names[] = {"PAUSED", "REC", "FAILED"};
    Serial.print("RECORDER 1 ");
    Serial.print(now);
    Serial.print(" ");
    Serial.print(names[state]);
    Serial.print(" ");
    Serial.print(state == RECORDING ? path : "-");
    Serial.print(" dropped=");
    Serial.println(dropped);
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
  startRecording();
}

void loop() {
  handleSerial();
  handleButton();
  updateLed();
  if (state == RECORDING && head - tail >= WRITE_BLOCK) {
    if (!writeBlock(WRITE_BLOCK)) fail();
  } else {
    delay(10);  // lets the CPU sleep between blocks
  }
}
