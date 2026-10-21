// Recorder firmware v3 for the Seeed XIAO nRF52840 Sense.
// Board package: "Seeed nRF52 Boards" (not the mbed-enabled one), board "Seeed XIAO nRF52840 Sense".
//
// Listens to the onboard PDM mic all the time and records to the microSD card only while
// there is sound: a recording starts when the level rises well above the room's noise floor
// (keeping the second before it), and ends after 10 s of quiet. Files are 16 kHz mono IMA ADPCM
// WAV (4:1, about 29 MB per hour of sound), at most 10 minutes each. The button pauses and
// resumes listening. Wiring (see firmware/README.md): SD on SPI (D8 SCK, D9 MISO, D10 MOSI,
// CS on D2), button between D1 and GND.
//
// Bluetooth LE ("Recorder"): the computer sets the clock, lists, downloads and deletes files,
// and can stream live audio (`recorder sync --ble`, `recorder live`, src/recorder/ble.py).
// It also reports the battery, as the standard battery service that phones and Windows show.
//
// Files are named by local start time, REC/20261020/093000.WAV (the SD library only does
// 8.3 names, so the date is the folder). Without a clock they go to REC/NOTIME/00000001.WAV.
// The board has no battery-backed clock: the computer sets it on every Bluetooth connection or
// over USB serial (`recorder settime`), and it holds while the power switch stays on.

#include <SPI.h>
#include <SD.h>
#include <PDM.h>
#include <bluefruit.h>
#include <time.h>
#include "adpcm.h"

const uint32_t SAMPLE_RATE = 16000;
const uint32_t BYTES_PER_SEC = SAMPLE_RATE * 2;    // in the ring buffer (16-bit)
const int FILE_BLOCK = 256;                        // ADPCM block in files
const uint32_t BLOCK_SAMPLES = 505;                // samples per file block
const uint32_t BLOCK_PCM = BLOCK_SAMPLES * 2;      // ring bytes per file block
const uint32_t CHUNK_SAMPLES = SAMPLE_RATE * 600;  // at most 10 minutes per file
const uint32_t FLUSH_SAMPLES = SAMPLE_RATE * 30;   // header + flush every 30 s: a power cut loses at most this
const int SD_CS = 2;                               // D2
const int BUTTON = 1;                              // D1
const time_t VALID_AFTER = 1700000000;  // anything earlier means the clock was never set

// Sound detection, on 16 ms frames. Tune with the level= and floor= that `?` reports.
const uint32_t FRAME = 512;                          // bytes: 256 samples
const uint32_t PREROLL_BYTES = BYTES_PER_SEC;        // keep 1 s before the sound that starts a recording
const uint32_t HANGOVER_BYTES = BYTES_PER_SEC * 10;  // end it after 10 s without sound
const float SOUND_RATIO = 3.0;  // a frame is sound when this many times above the noise floor...
const float MIN_LEVEL = 30;     // ...and at least this loud (mean distance from silence)
const int START_FRAMES = 5;     // sound frames among the last 16 that start a recording

// Battery: the XIAO measures it through a divider (1M / 510k) on PIN_VBAT, switched on by
// pulling VBAT_ENABLE low. Keep that pin low: high while charging can harm the input.
#ifndef PIN_VBAT
#define PIN_VBAT 32
#endif
#ifndef VBAT_ENABLE
#define VBAT_ENABLE 14
#endif
const float VBAT_SCALE = 3.0 / 4096 * (1000 + 510) / 510;
const int LOW_BATTERY = 15;  // percent

#ifndef LED_BLUE
#define LED_BLUE LED_BUILTIN
#endif

// Ring buffer between the PDM interrupt and everything else: 4 s of audio holds the pre-roll
// plus SD and Bluetooth stalls. Indices run freely; the size is a power of two. `scan` is how
// far sound detection has got; `tail` how far the SD writes have (or, while listening, the
// start of the pre-roll); `livePos` how far the live stream has.
const uint32_t RING_SIZE = 131072;
alignas(4) uint8_t ring[RING_SIZE];
volatile uint32_t head = 0, tail = 0, dropped = 0;
uint32_t scan = 0, lastSoundAt = 0, livePos = 0;
uint16_t recent = 0;  // one bit per recent frame, 1 = sound
float level = 0, noiseFloor = 0;
int16_t pdmBuf[256];
int16_t pcm[BLOCK_SAMPLES];  // one block being encoded

enum State { PAUSED, LISTENING, RECORDING, FAILED };
State state = PAUSED;
SDLib::File wav;
Adpcm fileEnc;
uint8_t out[16 * FILE_BLOCK];  // encoded blocks waiting for one SD write
uint32_t outLen = 0, dataBytes = 0, samples = 0, sinceFlush = 0, notimeIndex = 0;
char path[32];
uint32_t feedbackUntil = 0;
bool feedbackOn = false;
time_t clockBase = 0;  // the clock reads clockBase when millis() == clockMs
uint32_t clockMs = 0;
float vbat = 0;
int batteryPct = -1;

// ---- audio in ----

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

void peek(uint32_t from, void* dst, uint32_t n) {
  uint32_t pos = from & (RING_SIZE - 1);
  uint32_t first = min(n, RING_SIZE - pos);
  memcpy(dst, ring + pos, first);
  memcpy((uint8_t*)dst + first, ring, n - first);
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

// ---- clock and battery ----

time_t clockNow() { return clockBase + (time_t)((millis() - clockMs) / 1000); }

void setClock(time_t t) {
  clockBase = t;
  clockMs = millis();
}

void keepClock() {  // millis() wraps after 49 days: move the base along daily
  uint32_t secs = (millis() - clockMs) / 1000;
  if (secs > 86400) {
    clockBase += secs;
    clockMs += secs * 1000;
  }
}

bool clockValid() { return clockNow() > VALID_AFTER; }

void stamp(char* out, size_t size) {
  if (!clockValid()) {
    snprintf(out, size, "unset");
    return;
  }
  time_t t = clockNow();
  strftime(out, size, "%Y%m%dT%H%M%S", gmtime(&t));  // the clock holds local time
}

// Rough charge of a LiPo from its voltage (under the recorder's light load), in 5% steps.
int percentOf(float v) {
  const float volts[] = {3.27, 3.61, 3.69, 3.71, 3.73, 3.75, 3.77, 3.79, 3.80, 3.82, 3.84,
                         3.85, 3.87, 3.91, 3.95, 3.98, 4.02, 4.08, 4.11, 4.15, 4.20};
  if (v <= volts[0]) return 0;
  for (int i = 1; i < 21; i++)
    if (v < volts[i]) return (i - 1) * 5 + (int)(5 * (v - volts[i - 1]) / (volts[i] - volts[i - 1]));
  return 100;
}

BLEBas bas;

void readBattery() {
  uint32_t sum = 0;
  for (int i = 0; i < 8; i++) sum += analogRead(PIN_VBAT);
  vbat = sum / 8 * VBAT_SCALE;
  int pct = percentOf(vbat);
  if (pct != batteryPct) {
    batteryPct = pct;
    bas.write(pct);
    if (Bluefruit.connected()) bas.notify(pct);
  }
}

// ---- WAV files ----

void put32(SDLib::File& f, uint32_t v) { f.write((uint8_t*)&v, 4); }
void put16(SDLib::File& f, uint16_t v) { f.write((uint8_t*)&v, 2); }

const uint32_t HEADER_BYTES = 60;

void writeHeader(SDLib::File& f) {
  f.write((const uint8_t*)"RIFF", 4);
  put32(f, HEADER_BYTES - 8 + dataBytes);
  f.write((const uint8_t*)"WAVEfmt ", 8);
  put32(f, 20);
  put16(f, 0x11);  // IMA ADPCM
  put16(f, 1);     // mono
  put32(f, SAMPLE_RATE);
  put32(f, SAMPLE_RATE * FILE_BLOCK / BLOCK_SAMPLES);
  put16(f, FILE_BLOCK);
  put16(f, 4);
  put16(f, 2);
  put16(f, BLOCK_SAMPLES);
  f.write((const uint8_t*)"fact", 4);
  put32(f, 4);
  put32(f, samples);
  f.write((const uint8_t*)"data", 4);
  put32(f, dataBytes);
}

// The file for audio that started `back` seconds ago.
void nextPath(uint32_t back) {
  if (clockValid()) {
    time_t t = clockNow() - back;
    struct tm* g = gmtime(&t);
    char dir[16];
    snprintf(dir, sizeof(dir), "REC/%04d%02d%02d", (g->tm_year + 1900) % 10000, g->tm_mon + 1,
             g->tm_mday);
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
  nextPath((scan - tail) / BYTES_PER_SEC);  // what's still buffered started that long ago
  outLen = dataBytes = samples = sinceFlush = 0;
  SDLib::File f = SD.open(path, FILE_WRITE);
  if (!f) return false;
  writeHeader(f);
  f.close();
  wav = SD.open(path, O_READ | O_WRITE);  // not FILE_WRITE: that appends, and we rewrite the header
  if (!wav) return false;
  wav.seek(HEADER_BYTES);
  return true;
}

bool saveHeader() {
  if (outLen && wav.write(out, outLen) != outLen) return false;
  outLen = 0;
  wav.seek(0);
  writeHeader(wav);
  wav.seek(HEADER_BYTES + dataBytes);
  wav.flush();
  sinceFlush = 0;
  return true;
}

bool finishFile() {
  if (!wav) return true;
  bool ok = saveHeader();
  wav.close();
  return ok;
}

// Encodes the block in `pcm` into the file.
bool encodeBlock() {
  adpcmBlock(fileEnc, pcm, out + outLen, FILE_BLOCK, 0);
  outLen += FILE_BLOCK;
  dataBytes += FILE_BLOCK;
  samples += BLOCK_SAMPLES;
  sinceFlush += BLOCK_SAMPLES;
  if (outLen == sizeof(out)) {
    if (wav.write(out, outLen) != outLen) return false;
    outLen = 0;
  }
  if (sinceFlush >= FLUSH_SAMPLES && !saveHeader()) return false;
  if (samples >= CHUNK_SAMPLES) return finishFile() && startFile();
  return true;
}

// Encodes `n` (at most BLOCK_PCM) bytes from the ring into the file; a short last block is
// padded with its last sample.
bool writeBlock(uint32_t n) {
  peek(tail, pcm, n);
  tail += n;
  for (uint32_t i = n / 2; i < BLOCK_SAMPLES; i++) pcm[i] = i ? pcm[i - 1] : 0;
  return encodeBlock();
}

// Writes the buffered audio up to `end` and closes the file.
bool drainAndFinish(uint32_t end) {
  while (end != tail) {
    if (!writeBlock(min(end - tail, BLOCK_PCM))) return false;
  }
  return finishFile();
}

// ---- states ----

void fail() {
  PDM.end();
  if (wav) wav.close();
  state = FAILED;
}

void startListening() {
  head = tail = scan = livePos = 0;
  recent = 0;
  noiseFloor = 0;  // learn the room again
  if (!PDM.begin(1, SAMPLE_RATE)) return fail();
  state = LISTENING;
}

void pauseListening() {
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
    pauseListening();
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

// ---- Bluetooth ----
// Service 7f1e0001-5a1b-4c8e-9d2f-3b6a0c4e8d10 with three characteristics:
//   ...0002 CTRL  write a command line; replies come back as notifications, one line each
//   ...0003 DATA  notify: file contents, each packet = 4-byte offset (little-endian) + bytes
//   ...0004 LIVE  notify: live audio, each packet one ADPCM block (spare header byte = counter)

const uint8_t UUID_SERVICE[] = {0x10, 0x8D, 0x4E, 0x0C, 0x6A, 0x3B, 0x2F, 0x9D,
                                0x8E, 0x4C, 0x1B, 0x5A, 0x01, 0x00, 0x1E, 0x7F};
const uint8_t UUID_CTRL[] = {0x10, 0x8D, 0x4E, 0x0C, 0x6A, 0x3B, 0x2F, 0x9D,
                             0x8E, 0x4C, 0x1B, 0x5A, 0x02, 0x00, 0x1E, 0x7F};
const uint8_t UUID_DATA[] = {0x10, 0x8D, 0x4E, 0x0C, 0x6A, 0x3B, 0x2F, 0x9D,
                             0x8E, 0x4C, 0x1B, 0x5A, 0x03, 0x00, 0x1E, 0x7F};
const uint8_t UUID_LIVE[] = {0x10, 0x8D, 0x4E, 0x0C, 0x6A, 0x3B, 0x2F, 0x9D,
                             0x8E, 0x4C, 0x1B, 0x5A, 0x04, 0x00, 0x1E, 0x7F};
BLEService service(UUID_SERVICE);
BLECharacteristic ctrlChr(UUID_CTRL), dataChr(UUID_DATA), liveChr(UUID_LIVE);
uint16_t connHandle = BLE_CONN_HANDLE_INVALID;
uint16_t mtu = 23;

// Commands written over Bluetooth arrive in the Bluetooth task; the loop runs them.
char bleCmds[4][64];
volatile uint8_t bleIn = 0, bleDone = 0;

void onBleWrite(uint16_t, BLECharacteristic*, uint8_t* data, uint16_t len) {
  if ((uint8_t)(bleIn - bleDone) >= 4) return;  // the computer waits for replies: shouldn't happen
  char* cmd = bleCmds[bleIn % 4];
  len = min(len, (uint16_t)63);
  memcpy(cmd, data, len);
  cmd[len] = 0;
  while (len && (cmd[len - 1] == '\n' || cmd[len - 1] == '\r')) cmd[--len] = 0;
  bleIn++;
}

// A Print that sends each line as a CTRL notification.
class BleLines : public Print {
 public:
  size_t write(uint8_t c) override {
    if (c == '\n') {
      for (int i = 0; i < 5 && Bluefruit.connected() && !ctrlChr.notify(buf, len); i++) delay(5);
      len = 0;
    } else if (c != '\r' && len < sizeof(buf)) {
      buf[len++] = c;
    }
    return 1;
  }

 private:
  uint8_t buf[240];
  uint16_t len = 0;
};
BleLines bleLines;

// A transfer runs a little at a time from the loop, between the audio work.
enum Job { NO_JOB, LIST_JOB, GET_JOB };
Job job = NO_JOB;
SDLib::File jobDir, jobSub, jobFile;
char jobSubName[13];
uint32_t jobOff = 0;
uint8_t pkt[244];
uint16_t pktLen = 0;  // a DATA packet waiting to go out
bool liveOn = false;
uint8_t liveSeq = 0;
Adpcm liveEnc;

void endJob() {
  if (jobFile) jobFile.close();
  if (jobSub) jobSub.close();
  if (jobDir) jobDir.close();
  job = NO_JOB;
  pktLen = 0;
}

bool isOpenFile(const char* p) { return state == RECORDING && strcmp(p, path) == 0; }

// One step of a transfer; false when Bluetooth is busy and the loop should move on.
bool stepJob() {
  if (job == LIST_JOB) {
    if (!jobSub) {
      SDLib::File e = jobDir.openNextFile();
      if (!e) {
        bleLines.println("END");
        endJob();
      } else if (e.isDirectory()) {
        strncpy(jobSubName, e.name(), sizeof(jobSubName) - 1);
        jobSub = e;
      } else {
        e.close();
      }
      return true;
    }
    SDLib::File f = jobSub.openNextFile();
    if (!f) {
      jobSub.close();
      return true;
    }
    char p[40];
    snprintf(p, sizeof(p), "REC/%s/%s", jobSubName, f.name());
    bool file = !f.isDirectory();
    uint32_t size = f.size();
    f.close();
    if (file && !isOpenFile(p)) {  // the file being recorded comes next time
      bleLines.print("F ");
      bleLines.print(p);
      bleLines.print(" ");
      bleLines.println(size);
    }
    return true;
  }
  if (job == GET_JOB) {
    if (pktLen == 0) {
      int n = jobFile.read(pkt + 4, min(mtu - 7, 240));
      if (n <= 0) {
        bleLines.print("DONE ");
        bleLines.println(jobOff);
        endJob();
        return true;
      }
      memcpy(pkt, &jobOff, 4);
      pktLen = 4 + n;
      jobOff += n;
    }
    if (!dataChr.notify(pkt, pktLen)) return false;
    pktLen = 0;
    return true;
  }
  return false;
}

// Sends the live audio that's ready; false when there's none or Bluetooth is busy.
bool stepLive() {
  if (!liveOn || (state != LISTENING && state != RECORDING)) return false;
  int bytes = min(mtu - 3, 240);
  uint32_t need = (1 + (bytes - 4) * 2) * 2;
  if ((int32_t)(livePos - tail) < 0) livePos = tail;  // fell behind: skip ahead
  if (head - livePos < need) return false;
  uint8_t block[240];
  peek(livePos, pcm, need);
  adpcmBlock(liveEnc, pcm, block, bytes, liveSeq);
  if (!liveChr.notify(block, bytes)) return false;
  liveSeq++;
  livePos += need;
  return true;
}

void onConnect(uint16_t conn) {
  connHandle = conn;
  BLEConnection* c = Bluefruit.Connection(conn);
  c->requestPHY();
  c->requestDataLengthUpdate();
  c->requestMtuExchange(247);
  // Windows picks 30 ms otherwise, and three packets go out per interval: 24 KB/s. Ask for 7.5 ms
  // (the computer may settle on 15): more packets per second, for transfers and live audio.
  c->requestConnectionParameter(6);
}

void onDisconnect(uint16_t, uint8_t) {
  connHandle = BLE_CONN_HANDLE_INVALID;
  liveOn = false;
  endJob();
  mtu = 23;
}

void setupBle() {
  Bluefruit.configPrphBandwidth(BANDWIDTH_MAX);
  Bluefruit.begin();
  Bluefruit.autoConnLed(false);  // the LED is ours
  Bluefruit.setName("Recorder");
  Bluefruit.Periph.setConnectCallback(onConnect);
  Bluefruit.Periph.setDisconnectCallback(onDisconnect);
  bas.begin();
  service.begin();
  ctrlChr.setProperties(CHR_PROPS_WRITE | CHR_PROPS_WRITE_WO_RESP | CHR_PROPS_NOTIFY);
  ctrlChr.setPermission(SECMODE_OPEN, SECMODE_OPEN);
  ctrlChr.setMaxLen(244);
  ctrlChr.setWriteCallback(onBleWrite);
  ctrlChr.begin();
  dataChr.setProperties(CHR_PROPS_NOTIFY);
  dataChr.setPermission(SECMODE_OPEN, SECMODE_NO_ACCESS);
  dataChr.setMaxLen(244);
  dataChr.begin();
  liveChr.setProperties(CHR_PROPS_NOTIFY);
  liveChr.setPermission(SECMODE_OPEN, SECMODE_NO_ACCESS);
  liveChr.setMaxLen(244);
  liveChr.begin();
  Bluefruit.Advertising.addFlags(BLE_GAP_ADV_FLAGS_LE_ONLY_GENERAL_DISC_MODE);
  Bluefruit.Advertising.addService(service);
  Bluefruit.ScanResponse.addName();
  Bluefruit.Advertising.restartOnDisconnect(true);
  Bluefruit.Advertising.setInterval(32, 1600);  // fast for 30 s, then once a second (0.625 ms units)
  Bluefruit.Advertising.setFastTimeout(30);
  Bluefruit.Advertising.start(0);
}

// ---- button, LED, commands ----

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
// clock isn't set, long if the battery is low, blue instead of red while Bluetooth is connected),
// paused is dark, a failure (no card, card full) blinks fast.
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
    uint32_t p = ms % (state == RECORDING ? 1000 : 5000);  // long blips: the loop may be busy
    uint32_t blip = batteryPct >= 0 && batteryPct < LOW_BATTERY ? 400 : 150;
    on = p < blip || (!clockValid() && p >= blip + 250 && p < 2 * blip + 250);
  } else {
    on = false;
  }
  bool blue = Bluefruit.connected() && LED_BLUE != LED_BUILTIN;
  digitalWrite(LED_BUILTIN, on && !blue ? LOW : HIGH);
  if (LED_BLUE != LED_BUILTIN) digitalWrite(LED_BLUE, on && blue ? LOW : HIGH);
}

// One line per command, over USB serial or Bluetooth (src/recorder/clock.py, ble.py):
//   ?           -> RECORDER 3 <20261020T093000|unset> <LISTEN|REC|PAUSED|FAILED> <file|->
//                  dropped=<n> level=<now> floor=<noise floor> bat=<volts> pct=<percent>
//   T<seconds>  -> OK 20261020T093000   (local wall-clock time, counted as if it were UTC)
// Bluetooth only:
//   L                -> F <path> <size> per finished file, then END
//   G<offset> <path> -> the file from <offset> on DATA, then DONE <size>  (or ERR ...)
//   X                -> stops a listing or download: STOPPED
//   D<path>          -> OK, or ERR ...
//   S1 / S0          -> OK; live audio on LIVE starts / stops
void command(const char* line, Print& out, bool ble) {
  char stampNow[20];
  if (line[0] == 'T') {
    time_t t = (time_t)strtoull(line + 1, NULL, 10);
    if (t <= VALID_AFTER) {
      out.println("ERR bad time");
      return;
    }
    bool wasValid = clockValid();
    setClock(t);
    stamp(stampNow, sizeof(stampNow));
    out.print("OK ");  // answer first: switching files below takes a while
    out.println(stampNow);
    if (state == RECORDING && !wasValid) {  // go on in a properly named file
      if (!finishFile() || !startFile()) fail();
    }
  } else if (line[0] == '?') {
    readBattery();
    stamp(stampNow, sizeof(stampNow));
    const char* names[] = {"PAUSED", "LISTEN", "REC", "FAILED"};
    out.print("RECORDER 3 ");
    out.print(stampNow);
    out.print(" ");
    out.print(names[state]);
    out.print(" ");
    out.print(state == RECORDING ? path : "-");
    out.print(" dropped=");
    out.print(dropped);
    out.print(" level=");
    out.print(level, 0);
    out.print(" floor=");
    out.print(noiseFloor, 0);
    out.print(" bat=");
    out.print(vbat, 2);
    out.print(" pct=");
    out.println(batteryPct);
  } else if (!ble) {
    out.println("ERR bluetooth only");
  } else if (line[0] == 'X') {
    endJob();
    out.println("STOPPED");
  } else if (job != NO_JOB) {
    out.println("ERR busy");
  } else if (line[0] == 'L') {
    jobDir = SD.open("REC");
    if (!jobDir) {
      out.println("END");
      return;
    }
    job = LIST_JOB;
  } else if (line[0] == 'G') {
    char* rest;
    uint32_t off = strtoul(line + 1, &rest, 10);
    while (*rest == ' ') rest++;
    if (isOpenFile(rest)) {
      out.println("ERR recording");
      return;
    }
    jobFile = SD.open(rest, FILE_READ);
    if (!jobFile || !jobFile.seek(off)) {
      if (jobFile) jobFile.close();
      out.println("ERR no such file");
      return;
    }
    jobOff = off;
    job = GET_JOB;
  } else if (line[0] == 'D') {
    if (isOpenFile(line + 1)) {
      out.println("ERR recording");
    } else {
      out.println(SD.remove(line + 1) ? "OK" : "ERR not removed");
    }
  } else if (line[0] == 'S') {
    liveOn = line[1] == '1';
    livePos = head;
    out.println("OK");
  } else {
    out.println("ERR unknown command");
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
        command(line, Serial, false);
        len = 0;
      }
    } else if (len < (int)sizeof(line) - 1) {
      line[len++] = c;
    }
  }
}

void handleBle() {
  while (bleDone != bleIn) {
    command(bleCmds[bleDone % 4], bleLines, true);
    bleDone++;
  }
  if (connHandle != BLE_CONN_HANDLE_INVALID) {
    BLEConnection* c = Bluefruit.Connection(connHandle);
    if (c) mtu = c->getMtu();
  }
}

void setup() {
  Serial.begin(115200);  // doesn't wait for the computer: on battery there is none
  pinMode(LED_BUILTIN, OUTPUT);
  pinMode(LED_BLUE, OUTPUT);
  digitalWrite(LED_BUILTIN, HIGH);
  digitalWrite(LED_BLUE, HIGH);
  pinMode(BUTTON, INPUT_PULLUP);
  pinMode(VBAT_ENABLE, OUTPUT);
  digitalWrite(VBAT_ENABLE, LOW);
  analogReference(AR_INTERNAL_3_0);
  analogReadResolution(12);
  setupBle();
  readBattery();
  PDM.onReceive(onPDMdata);
  if (!SD.begin(SD_CS)) {
    state = FAILED;
    return;
  }
  startListening();
}

void loop() {
  static uint32_t batteryAt = 0;
  handleSerial();
  handleBle();
  handleButton();
  updateLed();
  keepClock();
  if (millis() - batteryAt > 60000) {
    batteryAt = millis();
    readBattery();
  }
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
  bool busy = false;
  if (state == RECORDING && scan - tail >= BLOCK_PCM) {  // writes never pass detection
    if (!writeBlock(BLOCK_PCM)) fail();
    busy = true;
  }
  for (int i = 0; i < 4 && stepLive(); i++) busy = true;
  for (int i = 0; i < 16 && job != NO_JOB && stepJob(); i++) busy = true;
  if (!busy) delay(10);  // lets the CPU sleep between blocks
}
