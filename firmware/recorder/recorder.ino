// Recorder firmware v4 for the Seeed XIAO nRF52840 Sense.
// Board package: "Seeed nRF52 Boards" (not the mbed-enabled one), board "Seeed XIAO nRF52840 Sense".
//
// Listens to the onboard PDM mic all the time and records to the microSD card only while
// there is sound: a recording starts when the level rises well above the room's noise floor
// (keeping the second before it), and ends after 10 s of quiet. Files are 16 kHz mono IMA ADPCM
// WAV (4:1, about 29 MB per hour of sound), at most 10 minutes each. The button pauses and
// resumes listening. Wiring (see firmware/README.md): SD on SPI (D8 SCK, D9 MISO, D10 MOSI,
// CS on D2), button between D1 and GND, optional 128x32 SSD1306 OLED on I2C (D4 SDA, D5 SCL).
//
// The OLED (when connected; USE_SCREEN below needs the Adafruit SSD1306 library) stays
// dark and lights for 5 s at power-on and when the button is pressed; a press while it is lit
// pauses or resumes. Without a screen, every press pauses or resumes.
//
// Holding the button for 2 s puts the recorder to sleep: no listening, no Bluetooth, LED and
// screen dark, the clock still running. A press wakes it. It sleeps instead of being switched
// off, so it needs no power switch and charges whenever it's plugged in.
//
// Bluetooth LE ("Recorder"): the computer sets the clock, lists, downloads and deletes files,
// and can stream live audio (`recorder sync --ble`, `recorder live`, src/recorder/ble.py).
// It also reports the battery, as the standard battery service that phones and Windows show.
//
// Files are named by local start time, REC/20261020/093000.WAV (the date is the folder). Without
// a clock they go to REC/NOTIME/00000001.WAV. The board has no battery-backed clock: the
// computer sets it on every Bluetooth connection or over USB serial (`recorder settime`), and it
// holds while the battery lasts, through a reset too (the reset button or the watchdog).
//
// The card is driven with SdFat (it comes with the board package): DMA transfers and
// multi-sector writes, so the card and the CPU are busy for a fraction of the time the Arduino
// SD library kept them. A watchdog restarts the recorder if it ever hangs.

#include <SPI.h>
#include <SdFat.h>
#include <PDM.h>
#include <bluefruit.h>
#include <time.h>
#include "adpcm.h"
// The OLED needs the Adafruit SSD1306 library (Library Manager). Without it, set this to 0.
// (A screen that isn't connected is fine either way: it's looked for at power-on.)
#ifndef USE_SCREEN
#define USE_SCREEN 1
#endif
#if USE_SCREEN
#include <Wire.h>
#include <Adafruit_SSD1306.h>
#define HAVE_SCREEN_LIB 1
#endif

const uint32_t SAMPLE_RATE = 16000;
const uint32_t BYTES_PER_SEC = SAMPLE_RATE * 2;    // in the ring buffer (16-bit)
const int FILE_BLOCK = 256;                        // ADPCM block in files
const uint32_t BLOCK_SAMPLES = 505;                // samples per file block
const uint32_t BLOCK_PCM = BLOCK_SAMPLES * 2;      // ring bytes per file block
const uint32_t CHUNK_SAMPLES = SAMPLE_RATE * 600;  // at most 10 minutes per file
const uint32_t FLUSH_SAMPLES = SAMPLE_RATE * 30;   // header + flush every 30 s: a power cut loses at most this
const int SD_CS = 2;                               // D2
const uint32_t SD_MHZ = 8;  // SPI clock for the card; lower it if a card misbehaves on long wires
const int BUTTON = 1;                              // D1
const time_t VALID_AFTER = 1700000000;  // anything earlier means the clock was never set
const uint32_t WATCHDOG_S = 10;         // restart if the loop stops for this long
const uint32_t CARD_RETRY_MS = 60000;   // after a card failure, try the card again this often

// The mic's gain, in the PDM's units: 0x28 is 0 dB, each step 0.5 dB. The library's default
// (20, -10 dB) left quiet speech only a few steps of the ADPCM coder above its smallest step.
const int PDM_GAIN = 0x28;

// Sound detection, on 16 ms frames. Tune with the level= and floor= that `?` reports.
const uint32_t FRAME = 512;                          // bytes: 256 samples
const uint32_t PREROLL_BYTES = BYTES_PER_SEC;        // keep 1 s before the sound that starts a recording
const uint32_t HANGOVER_BYTES = BYTES_PER_SEC * 10;  // end it after 10 s without sound
const float SOUND_RATIO = 3.0;  // a frame is sound when this many times above the noise floor...
const float MIN_LEVEL = 30;     // ...and at least this loud (mean distance from silence)
const int START_FRAMES = 5;     // sound frames among the last 16 that start a recording
const uint32_t MIN_SOUND_FRAMES = 40;  // a file with less sound than this (0.64 s) is deleted
                                       // when it ends: a knock or a door, not someone talking

// Battery: the XIAO measures it through a divider (1M / 510k) on PIN_VBAT, switched on by
// pulling VBAT_ENABLE low. Keep that pin low: high while charging can harm the input.
#ifndef PIN_VBAT
#define PIN_VBAT 32
#endif
#ifndef VBAT_ENABLE
#define VBAT_ENABLE 14
#endif
// The charger (BQ25101) charges at 50 mA unless P0.13 is pulled low, which makes it 100 mA:
// about 4-5 hours instead of 8-9 for a 400 mAh cell, and still gentle on it (0.25 C).
#ifndef PIN_CHARGING_CURRENT
#define PIN_CHARGING_CURRENT 22
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
uint32_t soundFrames = 0;  // in the file being recorded
float level = 0, noiseFloor = 0;
int16_t pdmBuf[256];
int16_t pcm[BLOCK_SAMPLES];  // one block being encoded

enum State { PAUSED, LISTENING, RECORDING, FAILED };
State state = PAUSED;
SdFat32 sd;
File32 wav;
Adpcm fileEnc;
// Encoded audio waiting for the card. It goes out in writes that end on a sector boundary, so
// the card gets whole sectors in one multi-sector write instead of a read-modify-write each.
const uint32_t SECTOR = 512;
const uint32_t WRITE_BYTES = 16 * SECTOR;
alignas(4) uint8_t out[WRITE_BYTES + FILE_BLOCK];
uint32_t outLen = 0, dataBytes = 0, samples = 0, sinceFlush = 0, notimeIndex = 0;
char path[40];
uint32_t feedbackUntil = 0, recordingSince = 0, failedAt = 0;
bool sleeping = false;
void noteRecording();  // for the screen's daily count
const uint32_t SLEEP_HOLD_MS = 2000;
bool feedbackOn = false;
time_t clockBase = 0;  // the clock reads clockBase when millis() == clockMs
uint32_t clockMs = 0;
float vbat = 0;
int batteryPct = -1;
uint32_t resetReason = 0;

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

bool clockValid() { return clockNow() > VALID_AFTER; }

// The clock survives a reset (the button, or the watchdog) in RAM that the startup code leaves
// alone: it's saved every second, and a reset loses about a second. Not after a power cut or a
// firmware upload (the bootloader runs in between, for who knows how long).
struct KeptClock {
  uint32_t magic, time, check;
};
KeptClock kept __attribute__((section(".noinit")));
const uint32_t KEPT_MAGIC = 0x52454331;  // "REC1"

void keepClock() {
  uint32_t secs = (millis() - clockMs) / 1000;
  if (secs > 86400) {  // millis() wraps after 49 days: move the base along daily
    clockBase += secs;
    clockMs += secs * 1000;
  }
  static uint32_t savedAt = 0;
  if (millis() - savedAt < 1000) return;
  savedAt = millis();
  if (!clockValid()) return;
  kept.time = (uint32_t)clockNow();
  kept.check = kept.time ^ KEPT_MAGIC;
  kept.magic = KEPT_MAGIC;
}

void restoreClock() {
  const uint32_t resets = POWER_RESETREAS_DOG_Msk | POWER_RESETREAS_LOCKUP_Msk |
                          POWER_RESETREAS_RESETPIN_Msk;
  bool ok = kept.magic == KEPT_MAGIC && kept.check == (kept.time ^ KEPT_MAGIC);
  if (ok && (resetReason & resets) && !(resetReason & POWER_RESETREAS_SREQ_Msk) &&
      (time_t)kept.time > VALID_AFTER)
    setClock((time_t)kept.time + 1);
  kept.magic = 0;
}

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

// ---- the card ----

bool cardBegin() { return sd.begin(SdSpiConfig(SD_CS, DEDICATED_SPI, SD_SCK_MHZ(SD_MHZ))); }

// Files get the recorder's time as their date (what a card reader shows).
void fileTime(uint16_t* date, uint16_t* time) {
  time_t t = clockValid() ? clockNow() : VALID_AFTER;
  struct tm* g = gmtime(&t);
  *date = FS_DATE(g->tm_year + 1900, g->tm_mon + 1, g->tm_mday);
  *time = FS_TIME(g->tm_hour, g->tm_min, g->tm_sec);
}

// recorder-device.json at the card's root marks it as a recorder for `recorder sync` with a
// card reader; written once, if it isn't there.
void writeMarker() {
  const char* name = "recorder-device.json";
  if (sd.exists(name)) return;
  uint8_t mac[6];
  Bluefruit.getAddr(mac);
  char text[48];
  int n = snprintf(text, sizeof text, "{\"id\": \"recorder-%02x%02x\"}\n", mac[1], mac[0]);
  File32 f = sd.open(name, O_WRONLY | O_CREAT | O_TRUNC);
  if (!f) return;
  f.write(text, n);
  f.close();
}

// ---- WAV files ----

const uint32_t HEADER_BYTES = 60;

void put32(uint8_t*& p, uint32_t v) {
  memcpy(p, &v, 4);
  p += 4;
}
void put16(uint8_t*& p, uint16_t v) {
  memcpy(p, &v, 2);
  p += 2;
}
void putTag(uint8_t*& p, const char* tag) {
  memcpy(p, tag, 4);
  p += 4;
}

void writeHeader(uint8_t* h) {
  uint8_t* p = h;
  putTag(p, "RIFF");
  put32(p, HEADER_BYTES - 8 + dataBytes);
  putTag(p, "WAVE");
  putTag(p, "fmt ");
  put32(p, 20);
  put16(p, 0x11);  // IMA ADPCM
  put16(p, 1);     // mono
  put32(p, SAMPLE_RATE);
  put32(p, SAMPLE_RATE * FILE_BLOCK / BLOCK_SAMPLES);
  put16(p, FILE_BLOCK);
  put16(p, 4);
  put16(p, 2);
  put16(p, BLOCK_SAMPLES);
  putTag(p, "fact");
  put32(p, 4);
  put32(p, samples);
  putTag(p, "data");
  put32(p, dataBytes);
}

// The file for audio that started `back` seconds ago.
void nextPath(uint32_t back) {
  if (clockValid()) {
    time_t t = clockNow() - back;
    struct tm* g = gmtime(&t);
    char dir[24];
    snprintf(dir, sizeof(dir), "REC/%04d%02d%02d", (g->tm_year + 1900) % 10000,
             (g->tm_mon + 1) % 100, g->tm_mday % 100);
    sd.mkdir(dir);
    snprintf(path, sizeof(path), "%s/%02d%02d%02d.WAV", dir, g->tm_hour % 100, g->tm_min % 100,
             g->tm_sec % 100);
    for (char c = 'A'; sd.exists(path) && c <= 'Z'; c++)  // same second again: 093000A.WAV
      snprintf(path, sizeof(path), "%s/%02d%02d%02d%c.WAV", dir, g->tm_hour % 100,
               g->tm_min % 100, g->tm_sec % 100, c);
    return;
  }
  sd.mkdir("REC/NOTIME");
  do {
    snprintf(path, sizeof(path), "REC/NOTIME/%08lu.WAV", (unsigned long)++notimeIndex);
  } while (sd.exists(path));
}

bool startFile() {
  nextPath((scan - tail) / BYTES_PER_SEC);  // what's still buffered started that long ago
  outLen = dataBytes = samples = sinceFlush = soundFrames = 0;
  if (!wav.open(path, O_RDWR | O_CREAT | O_TRUNC)) return false;
  uint8_t h[HEADER_BYTES];
  writeHeader(h);
  return wav.write(h, HEADER_BYTES) == HEADER_BYTES;
}

// Sends the waiting audio to the card: up to the last whole sector, or (`all`) everything.
bool writeOut(bool all) {
  uint32_t at = HEADER_BYTES + dataBytes - outLen;  // where out[0] goes in the file
  uint32_t n = all ? outLen : (at + outLen) / SECTOR * SECTOR - at;
  if (n == 0 || n > outLen) return true;
  if (wav.write(out, n) != n) return false;
  outLen -= n;
  memmove(out, out + n, outLen);
  return true;
}

bool saveHeader() {
  if (!writeOut(true)) return false;
  uint8_t h[HEADER_BYTES];
  writeHeader(h);
  if (!wav.seekSet(0) || wav.write(h, HEADER_BYTES) != HEADER_BYTES) return false;
  if (!wav.seekSet(HEADER_BYTES + dataBytes) || !wav.sync()) return false;
  sinceFlush = 0;
  return true;
}

bool finishFile() {
  if (!wav) return true;
  bool ok = saveHeader();
  ok = wav.close() && ok;
  return ok;
}

// Encodes the block in `pcm` into the file.
bool encodeBlock() {
  adpcmBlock(fileEnc, pcm, out + outLen, FILE_BLOCK, 0);
  outLen += FILE_BLOCK;
  dataBytes += FILE_BLOCK;
  samples += BLOCK_SAMPLES;
  sinceFlush += BLOCK_SAMPLES;
  if (outLen >= WRITE_BYTES && !writeOut(false)) return false;
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
  if (wav) {
    bool empty = dataBytes == 0;
    wav.close();
    if (empty) sd.remove(path);  // a full card: don't leave a file per attempt
  }
  state = FAILED;
  failedAt = millis();
}

void startListening() {
  head = tail = scan = livePos = 0;
  recent = 0;
  noiseFloor = 0;  // learn the room again
  if (!PDM.begin(1, SAMPLE_RATE)) return fail();
  PDM.setGain(PDM_GAIN);
  state = LISTENING;
}

void pauseListening() {
  PDM.end();
  if (state == RECORDING) {
    if (!drainAndFinish(head)) return fail();
    noteRecording();
  }
  state = PAUSED;
}

void startSegment() {
  if (!startFile()) return fail();
  soundFrames = __builtin_popcount(recent);
  lastSoundAt = scan;
  recordingSince = millis();
  state = RECORDING;
}

void endSegment() {
  bool keep = soundFrames >= MIN_SOUND_FRAMES;
  if (!drainAndFinish(scan)) return fail();  // the rest is the next pre-roll
  if (!keep) sd.remove(path);
  else noteRecording();
  recent = 0;
  state = LISTENING;
}

// After a failure: the card again (taken out and put back, or a one-off error), then listening.
bool retryCard() {
  sd.end();
  if (!cardBegin()) {
    failedAt = millis();
    return false;
  }
  startListening();
  return state != FAILED;
}

void onPress() {
  if (state == LISTENING || state == RECORDING) {
    pauseListening();
  } else if (state == FAILED) {
    retryCard();  // card back in? start over
  } else {
    startListening();
  }
  feedbackOn = state != PAUSED;
  feedbackUntil = millis() + 600;
}

// ---- watchdog ----
// Restarts the recorder if the loop stops being called (a card or radio that never answers).
// It can't be stopped once started, not even by a reset other than a power cut; the bootloader
// keeps it fed while it waits for an upload.

void startWatchdog() {
  if (!(NRF_WDT->RUNSTATUS & 1)) {
    NRF_WDT->CONFIG = (WDT_CONFIG_SLEEP_Run << WDT_CONFIG_SLEEP_Pos) |
                      (WDT_CONFIG_HALT_Pause << WDT_CONFIG_HALT_Pos);
    NRF_WDT->CRV = 32768 * WATCHDOG_S - 1;
    NRF_WDT->RREN = WDT_RREN_RR0_Msk;
    NRF_WDT->TASKS_START = 1;
  }
}

void feedWatchdog() { NRF_WDT->RR[0] = WDT_RR_RR_Reload; }

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
volatile uint32_t links = 0;  // connections so far: a transfer belongs to one of them
bool bleOk = false;
// Notifications the radio can hold: with 3 it sends at most 3 packets per connection event and
// then waits for the loop to queue more. 16 lets it fill the event. (If the SoftDevice can't
// fit that in its RAM, setupBle falls back to the usual 3.)
const uint8_t TX_QUEUE = 16;
uint8_t txQueue = 0;

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
      // notify() waits up to 100 ms for room in the queue, which a transfer may have filled
      for (int i = 0; i < 10 && Bluefruit.connected() && !ctrlChr.notify(buf, len); i++) delay(5);
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
uint32_t jobLink = 0;
File32 jobDir, jobSub, jobFile;
char jobSubName[16];
uint32_t jobOff = 0;  // file offset of the next packet
// A download reads the card in large pieces (one multi-sector read) and sends them as packets.
alignas(4) uint8_t jobBuf[8 * SECTOR];
uint32_t jobBufLen = 0, jobBufAt = 0;
bool jobEnd = false;
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

bool isOpenFile(const char* p) { return state == RECORDING && strcasecmp(p, path) == 0; }

// The next piece of the file being sent, read so that it ends on a sector boundary.
bool fillJobBuf() {
  uint32_t left = jobBufLen - jobBufAt;
  memmove(jobBuf, jobBuf + jobBufAt, left);
  jobBufLen = left;
  jobBufAt = 0;
  uint32_t at = jobOff + left;  // file offset of the first byte to read
  uint32_t want = sizeof(jobBuf) - left;
  if (want > SECTOR) want -= (at + want) % SECTOR;
  int n = jobFile.read(jobBuf + left, want);
  if (n < 0) return false;
  if ((uint32_t)n < want) jobEnd = true;
  jobBufLen += n;
  return true;
}

// One step of a transfer; false when Bluetooth is busy and the loop should move on.
bool stepJob() {
  if (job == LIST_JOB) {
    if (!jobSub) {
      File32 e = jobDir.openNextFile();
      if (!e) {
        bleLines.println("END");
        endJob();
      } else if (e.isDir() && !e.isHidden()) {
        e.getName(jobSubName, sizeof(jobSubName));
        jobSub = e;
      } else {
        e.close();
      }
      return true;
    }
    File32 f = jobSub.openNextFile();
    if (!f) {
      jobSub.close();
      return true;
    }
    char name[16], p[48];
    f.getName(name, sizeof(name));
    snprintf(p, sizeof(p), "REC/%s/%s", jobSubName, name);
    bool file = !f.isDir() && !f.isHidden() && name[0] != '.';
    uint32_t size = f.fileSize();
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
      uint32_t room = min(mtu - 7, 240);
      if (jobBufLen - jobBufAt < room && !jobEnd && !fillJobBuf()) {
        bleLines.println("ERR read failed");
        endJob();
        return true;
      }
      uint32_t n = min(room, jobBufLen - jobBufAt);
      if (n == 0) {
        bleLines.print("DONE ");
        bleLines.println(jobOff);
        endJob();
        return true;
      }
      memcpy(pkt, &jobOff, 4);
      memcpy(pkt + 4, jobBuf + jobBufAt, n);
      jobBufAt += n;
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
  // Audio behind `tail` stays in the ring until the microphone comes round to it again, so the
  // stream can lag the SD writes; only when it's about to be overwritten does it skip ahead
  // (counting the blocks skipped, so the computer sees them as lost).
  const uint32_t keep = RING_SIZE - 8192;
  if (head - livePos > keep) {
    uint32_t skip = (head - livePos - keep + need - 1) / need;
    livePos += skip * need;
    liveSeq += skip;
  }
  if (head - livePos < need) return false;
  uint8_t block[240];
  peek(livePos, pcm, need);
  adpcmBlock(liveEnc, pcm, block, bytes, liveSeq);
  if (!liveChr.notify(block, bytes)) return false;
  liveSeq++;
  livePos += need;
  return true;
}

// The connection interval: 7.5 ms while sending (more packets per second) and 100-200 ms when
// idle for a while, which costs the radio far less while `recorder ui` stays connected.
const uint32_t RELAX_AFTER_MS = 15000;
bool linkRelaxed = false;
uint32_t linkBusyAt = 0;

void requestInterval(bool fast) {
  if (fast) {
    // Windows picks 30 ms otherwise, and three packets go out per interval: 24 KB/s. Ask for 7.5
    // ms (the computer may settle on 15): more packets per second, for transfers and live audio.
    Bluefruit.Connection(connHandle)->requestConnectionParameter(6);
  } else {
    ble_gap_conn_params_t p = {80, 160, 0, 400};  // 100-200 ms (1.25 ms units), 4 s timeout
    sd_ble_gap_conn_param_update(connHandle, &p);
  }
  linkRelaxed = !fast;
}

void tuneLink() {
  if (connHandle == BLE_CONN_HANDLE_INVALID || !Bluefruit.Connection(connHandle)) return;
  bool busy = job != NO_JOB || liveOn;
  if (busy) linkBusyAt = millis();
  if (busy && linkRelaxed) requestInterval(true);
  if (!busy && !linkRelaxed && millis() - linkBusyAt > RELAX_AFTER_MS) requestInterval(false);
}

void onConnect(uint16_t conn) {
  connHandle = conn;
  links++;
  linkBusyAt = millis();
  BLEConnection* c = Bluefruit.Connection(conn);
  c->requestPHY(BLE_GAP_PHY_2MBPS);
  c->requestDataLengthUpdate();
  c->requestMtuExchange(247);
  requestInterval(true);
}

// In the Bluetooth task: the loop notices (by `links` and connHandle) and ends what was running.
void onDisconnect(uint16_t, uint8_t) {
  connHandle = BLE_CONN_HANDLE_INVALID;
  mtu = 23;
}

void setupBle() {
  const uint8_t queues[] = {TX_QUEUE, 3};
  for (uint8_t q : queues) {
    Bluefruit.configPrphConn(247, 100, q, BLE_GATTC_WRITE_CMD_TX_QUEUE_SIZE_DEFAULT);
    if (Bluefruit.begin()) {
      txQueue = q;
      break;
    }
    sd_softdevice_disable();  // its RAM is too small for that queue: try again with less
  }
  if (!txQueue) return;  // no Bluetooth; recording works without it
  bleOk = true;
  sd_power_dcdc_mode_set(NRF_POWER_DCDC_ENABLE);  // the XIAO has the DC/DC's inductor: ~40% less current
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

// ---- screen ----

const uint32_t SCREEN_MS = 5000;
const char* OWNER = "Bo";             // greeted at power-on and on waking; "" for no name
const uint32_t HELLO_MS = 1600;       // the hello animation at power-on and on waking
uint32_t helloUntil = 0;
uint8_t waveHistory[48];              // the sound level, one column every 100 ms
uint8_t waveAt = 0;
#ifdef HAVE_SCREEN_LIB
Adafruit_SSD1306 screen(128, 32, &Wire, -1);
#endif
bool haveScreen = false, screenShowing = false;
uint32_t screenUntil = 0, screenDrawnAt = 0;

void setupScreen() {
#ifdef HAVE_SCREEN_LIB
  Wire.begin();
  Wire.beginTransmission(0x3C);
  if (Wire.endTransmission() != 0) return;  // nothing connected
  haveScreen = screen.begin(SSD1306_SWITCHCAPVCC, 0x3C);
  screenUntil = millis() + HELLO_MS + SCREEN_MS;
  helloUntil = millis() + HELLO_MS;
  screenDrawnAt = millis() - 1000;
#endif
}

bool screenOn() { return haveScreen && (int32_t)(screenUntil - millis()) > 0; }

void showScreen() {
  screenUntil = millis() + SCREEN_MS;
  screenDrawnAt = millis() - 1000;  // draw now
}

// Recordings kept since midnight (or since power-on, without a clock), for the screen.
uint16_t clipsToday = 0;
uint32_t secondsToday = 0;
int statsDay = -1;

int today() {
  if (!clockValid()) return -2;
  return (int)(clockNow() / 86400);
}

void newDayStats() {
  if (today() == statsDay) return;
  statsDay = today();
  clipsToday = 0;
  secondsToday = 0;
}

void noteRecording() {
  newDayStats();
  clipsToday++;
  secondsToday += (millis() - recordingSince) / 1000;
}

#ifdef HAVE_SCREEN_LIB
void drawBluetooth(int x, int y) {  // the rune, 5 x 8
  screen.drawLine(x + 2, y, x + 2, y + 7, SSD1306_WHITE);
  screen.drawLine(x + 2, y, x + 4, y + 2, SSD1306_WHITE);
  screen.drawLine(x + 4, y + 2, x, y + 5, SSD1306_WHITE);
  screen.drawLine(x + 2, y + 7, x + 4, y + 5, SSD1306_WHITE);
  screen.drawLine(x + 4, y + 5, x, y + 2, SSD1306_WHITE);
}

void drawBattery(int x, int y, int pct) {  // 16 x 8
  screen.drawRect(x, y, 14, 8, SSD1306_WHITE);
  screen.fillRect(x + 14, y + 2, 2, 4, SSD1306_WHITE);
  if (pct > 0) screen.fillRect(x + 2, y + 2, max(1, pct * 10 / 100), 4, SSD1306_WHITE);
}

// The level of the last few seconds, scrolling left: each column is how far the sound was
// above the room's noise, mirrored around the middle line. A recording starts at about
// half height.
void drawWave(int x, int mid) {
  for (int i = 0; i < 48; i += 2) {
    int h = waveHistory[(waveAt + i) % 48];
    screen.drawFastVLine(x + i, mid - h, 2 * h + 1, SSD1306_WHITE);
  }
}

// Power-on and waking: sound bars rise in a wave, then the greeting types itself out.
void drawHello(uint32_t t) {
  for (int i = 0; i < 9; i++) {
    float phase = t / 90.0f - i * 0.7f;
    int h = phase < 0 ? 0 : (int)(3 + 4 * fabsf(sinf(phase)) * min(1.0f, phase / 3));
    screen.fillRect(2 + i * 4, 16 - h, 2, 2 * h, SSD1306_WHITE);
  }
  char text[24];
  const char* hello = "Hello";
  if (clockValid()) {
    time_t now = clockNow();
    int hour = gmtime(&now)->tm_hour;
    hello = hour < 5 ? "Hi" : hour < 12 ? "Morning" : hour < 18 ? "Hi" : "Evening";
  }
  snprintf(text, sizeof text, OWNER[0] ? "%s, %s" : "%s", hello, OWNER);
  int shown = t < 300 ? 0 : min((int)strlen(text), (int)((t - 300) / 60));
  text[shown] = 0;
  screen.setTextSize(1);
  screen.setCursor(42, 4);
  screen.print(text);
  if (t > 900) {
    screen.setCursor(42, 18);
    screen.print("listening...");
  }
  if ((t / 250) % 2 && shown < (int)strlen(text) + 1) screen.fillRect(42 + shown * 6, 4, 5, 8, SSD1306_WHITE);
}

#endif

// Going to sleep: a moon and some Zs floating up, with the LED's three slow blinks.
void playGoodnight() {
  for (uint32_t t = 0; t < 1200; t += 50) {
    digitalWrite(LED_BUILTIN, (t / 200) % 2 ? HIGH : LOW);
#ifdef HAVE_SCREEN_LIB
    if (haveScreen) {
      screen.clearDisplay();
      screen.fillCircle(12, 16, 9, SSD1306_WHITE);  // a crescent: a circle with a bite
      screen.fillCircle(17, 13, 8, SSD1306_BLACK);
      screen.setTextSize(1);
      screen.setCursor(34, 12);
      screen.print("Good night");
      for (int z = 0; z < 3; z++) {
        int age = (int)t - z * 300;
        if (age < 0) continue;
        screen.setTextSize(z == 2 ? 2 : 1);
        screen.setCursor(100 + z * 8, 22 - age / 60 - z * 4);
        screen.print("z");
      }
      screen.display();
      screen.ssd1306_command(SSD1306_DISPLAYON);
      screenShowing = true;
    }
#endif
    delay(50);
  }
  digitalWrite(LED_BUILTIN, HIGH);
}

// Every 100 ms, whether the screen is on or not, so the wave has a past when it lights up.
void sampleWave() {
  static uint32_t at = 0;
  if (millis() - at < 100) return;
  at = millis();
  float ratio = level / max(noiseFloor, 1.0f);
  int h = (state == LISTENING || state == RECORDING) && ratio > 1 ? (int)(log2f(ratio) * 2.5f) : 0;
  waveHistory[waveAt] = min(h, 7);
  waveAt = (waveAt + 1) % 48;
}

//   21:11              BT  85% [###]
//   (o) 1:23                ▁▂▃▅▆     REC with time and sound meter / LISTEN / PAUSED / NO CARD
//   Today 4 clips 12 min
void updateScreen() {
#ifdef HAVE_SCREEN_LIB
  if (!haveScreen) return;
  sampleWave();
  if (!screenOn()) {
    if (screenShowing) screen.ssd1306_command(SSD1306_DISPLAYOFF);  // about 10 uA when off
    screenShowing = false;
    return;
  }
  bool hello = (int32_t)(helloUntil - millis()) > 0;
  if (screenShowing && millis() - screenDrawnAt < (hello ? 40u : 100u)) return;
  screenDrawnAt = millis();
  char line[32];
  screen.clearDisplay();
  screen.setTextColor(SSD1306_WHITE);
  if (hello) {
    drawHello(HELLO_MS - (helloUntil - millis()));
    screen.display();
    if (!screenShowing) screen.ssd1306_command(SSD1306_DISPLAYON);
    screenShowing = true;
    return;
  }
  screen.setTextSize(1);

  screen.setCursor(0, 0);
  if (clockValid()) {
    time_t t = clockNow();
    strftime(line, sizeof line, "%H:%M", gmtime(&t));  // the clock holds local time
    screen.print(line);
  } else {
    screen.print("--:--");
  }
  bool busyBt = job != NO_JOB || liveOn;  // blinks while sending files or live audio
  if (Bluefruit.connected() && (!busyBt || (millis() / 150) % 2)) drawBluetooth(72, 0);
  if (batteryPct >= 0) {
    snprintf(line, sizeof line, "%3d%%", batteryPct);
    screen.setCursor(84, 0);
    screen.print(line);
    drawBattery(111, 0, batteryPct);
  }

  screen.setTextSize(2);
  if (state == RECORDING) {
    if ((millis() / 500) % 2) screen.fillCircle(5, 17, 5, SSD1306_WHITE);
    else screen.drawCircle(5, 17, 5, SSD1306_WHITE);
    uint32_t sec = (millis() - recordingSince) / 1000;
    snprintf(line, sizeof line, "%lu:%02lu", (unsigned long)(sec / 60), (unsigned long)(sec % 60));
    screen.setCursor(15, 10);
    screen.print(line);
  } else if (state == LISTENING) {
    screen.setCursor(0, 10);
    screen.print("LISTEN");
  } else if (state == PAUSED) {
    screen.fillRect(1, 11, 3, 12, SSD1306_WHITE);
    screen.fillRect(7, 11, 3, 12, SSD1306_WHITE);
    screen.setCursor(15, 10);
    screen.print("PAUSED");
  } else {
    screen.setCursor(0, 10);
    screen.print("NO CARD");
  }
  if (state == RECORDING || state == LISTENING) drawWave(80, 16);

  screen.setTextSize(1);
  screen.setCursor(0, 25);
  if (state == FAILED) {
    screen.print("Press to retry");
  } else if (!clockValid()) {
    screen.print("Clock not set");
  } else {
    newDayStats();
    snprintf(line, sizeof line, "Today %u clip%s %lu min", clipsToday, clipsToday == 1 ? "" : "s",
             (unsigned long)((secondsToday + 30) / 60));
    screen.print(line);
  }
  screen.display();
  if (!screenShowing) screen.ssd1306_command(SSD1306_DISPLAYON);
  screenShowing = true;
#endif
}

// ---- button, LED, commands ----

void goToSleep() {
  if (state == LISTENING || state == RECORDING) pauseListening();  // closes the file
  playGoodnight();  // three slow blinks, and the moon on the screen
  endJob();
  liveOn = false;
  if (bleOk) {
    Bluefruit.Advertising.restartOnDisconnect(false);
    if (connHandle != BLE_CONN_HANDLE_INVALID) Bluefruit.disconnect(connHandle);
    Bluefruit.Advertising.stop();
  }
  screenUntil = millis();
  sleeping = true;
}

void wakeUp() {
  sleeping = false;
  if (bleOk) {
    Bluefruit.Advertising.restartOnDisconnect(true);
    Bluefruit.Advertising.start(0);
  }
  readBattery();
  onPress();  // starts listening (or retries the card), with the usual LED feedback
  if (haveScreen) {
    showScreen();
    screenUntil += HELLO_MS;
    helloUntil = millis() + HELLO_MS;
  }
}

// Acts on release for a short press, so that holding for 2 s can mean sleep instead.
void handleButton() {
  static bool last = HIGH, stable = HIGH, held = false;
  static uint32_t changedAt = 0, pressedAt = 0;
  bool now = digitalRead(BUTTON);
  if (now != last) {
    last = now;
    changedAt = millis();
  }
  if (millis() - changedAt > 30 && now != stable) {
    stable = now;
    if (stable == LOW) {
      pressedAt = millis();
      held = false;
      if (sleeping) {
        wakeUp();
        held = true;  // this press is used up
      }
    } else if (!held) {
      if (haveScreen && !screenOn()) {
        showScreen();  // the first press only shows the screen
      } else {
        onPress();
        if (haveScreen) showScreen();
      }
    }
  }
  if (stable == LOW && !held && millis() - pressedAt >= SLEEP_HOLD_MS) {
    held = true;
    goToSleep();
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

const char* resetName() {
  if (resetReason & POWER_RESETREAS_DOG_Msk) return "watchdog";
  if (resetReason & POWER_RESETREAS_LOCKUP_Msk) return "lockup";
  if (resetReason & POWER_RESETREAS_SREQ_Msk) return "soft";
  if (resetReason & POWER_RESETREAS_RESETPIN_Msk) return "button";
  return "power";
}

// One line per command, over USB serial or Bluetooth (src/recorder/clock.py, ble.py):
//   ?           -> RECORDER 4 <20261020T093000|unset> <LISTEN|REC|PAUSED|FAILED> <file|->
//                  dropped=<n> level=<now> floor=<noise floor> bat=<volts> pct=<percent>
//                  screen=<ok|none|nolib> up=<seconds> reset=<power|button|watchdog|...>
//                  txq=<notification queue> ci=<connection interval, ms>
//   T<seconds>  -> OK 20261020T093000   (local wall-clock time, counted as if it were UTC)
// Bluetooth only:
//   L                -> F <path> <size> per finished file, then END
//   G<offset> <path> -> the file from <offset> on DATA, then DONE <size>  (or ERR ...)
//   X                -> stops a listing or download: STOPPED
//   D<path>          -> OK, or ERR ...  (an emptied date folder goes too)
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
    out.print("RECORDER 4 ");
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
    out.print(batteryPct);
    out.print(" screen=");  // ok, none (not found at power-on) or nolib (library not installed)
#ifdef HAVE_SCREEN_LIB
    out.print(haveScreen ? "ok" : "none");
#else
    out.print("nolib");
#endif
    out.print(" up=");
    out.print(millis() / 1000);
    out.print(" reset=");
    out.print(resetName());
    out.print(" txq=");
    out.print(txQueue);
    BLEConnection* c = connHandle != BLE_CONN_HANDLE_INVALID ? Bluefruit.Connection(connHandle) : NULL;
    out.print(" ci=");
    out.println(c ? c->getConnectionInterval() * 1.25f : 0, 1);
  } else if (!ble) {
    out.println("ERR bluetooth only");
  } else if (line[0] == 'X') {
    endJob();
    out.println("STOPPED");
  } else if (job != NO_JOB) {
    out.println("ERR busy");
  } else if (line[0] == 'L') {
    jobDir = sd.open("REC");
    if (!jobDir) {
      out.println("END");
      return;
    }
    job = LIST_JOB;
    jobLink = links;
  } else if (line[0] == 'G') {
    char* rest;
    uint32_t off = strtoul(line + 1, &rest, 10);
    while (*rest == ' ') rest++;
    if (isOpenFile(rest)) {
      out.println("ERR recording");
      return;
    }
    jobFile = sd.open(rest, O_RDONLY);
    if (!jobFile || jobFile.isDir() || off > jobFile.fileSize() || !jobFile.seekSet(off)) {
      if (jobFile) jobFile.close();
      out.println("ERR no such file");
      return;
    }
    jobOff = off;
    jobBufLen = jobBufAt = 0;
    jobEnd = false;
    job = GET_JOB;
    jobLink = links;
  } else if (line[0] == 'D') {
    const char* p = line + 1;
    if (isOpenFile(p)) {
      out.println("ERR recording");
    } else if (!sd.remove(p)) {
      out.println("ERR not removed");
    } else {
      char dir[32];
      const char* slash = strrchr(p, '/');
      if (slash && slash - p < (int)sizeof(dir)) {
        memcpy(dir, p, slash - p);
        dir[slash - p] = 0;
        if (strcasecmp(dir, "REC") != 0) sd.rmdir(dir);  // only goes if it's empty now
      }
      out.println("OK");
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
  if (!bleOk) return;
  // A transfer or live stream of a connection that has ended (or been replaced) stops here,
  // not in the disconnect callback: that runs in the Bluetooth task, maybe mid-read of the card.
  bool linked = connHandle != BLE_CONN_HANDLE_INVALID;
  if (job != NO_JOB && (!linked || jobLink != links)) endJob();
  if (!linked) {
    liveOn = false;
    linkRelaxed = false;
  }
  while (bleDone != bleIn) {
    command(bleCmds[bleDone % 4], bleLines, true);
    bleDone++;
  }
  if (linked) {
    BLEConnection* c = Bluefruit.Connection(connHandle);
    if (c) mtu = c->getMtu();
    tuneLink();
  }
}

void setup() {
  resetReason = readResetReason();
  startWatchdog();
  Serial.begin(115200);  // doesn't wait for the computer: on battery there is none
  pinMode(LED_BUILTIN, OUTPUT);
  pinMode(LED_BLUE, OUTPUT);
  digitalWrite(LED_BUILTIN, HIGH);
  digitalWrite(LED_BLUE, HIGH);
  pinMode(BUTTON, INPUT_PULLUP);
  pinMode(VBAT_ENABLE, OUTPUT);
  digitalWrite(VBAT_ENABLE, LOW);
  pinMode(PIN_CHARGING_CURRENT, OUTPUT);
  digitalWrite(PIN_CHARGING_CURRENT, LOW);  // charge at 100 mA
#ifdef PIN_LSM6DS3TR_C_POWER
  pinMode(PIN_LSM6DS3TR_C_POWER, OUTPUT);  // the motion sensor isn't used: unpowered
  digitalWrite(PIN_LSM6DS3TR_C_POWER, LOW);
#endif
  analogReference(AR_INTERNAL_3_0);
  analogReadResolution(12);
  restoreClock();
  setupBle();
  readBattery();
  setupScreen();
  FsDateTime::setCallback(fileTime);
  PDM.onReceive(onPDMdata);
  if (!cardBegin()) {
    state = FAILED;
    failedAt = millis();
    return;
  }
  writeMarker();
  startListening();
}

void loop() {
  static uint32_t batteryAt = 0;
  feedWatchdog();
  if (sleeping) {
    handleSerial();
    handleButton();
    updateScreen();  // turns it off
    keepClock();
    digitalWrite(LED_BUILTIN, HIGH);
    if (LED_BLUE != LED_BUILTIN) digitalWrite(LED_BLUE, HIGH);
    delay(50);
    return;
  }
  handleSerial();
  handleBle();
  handleButton();
  updateLed();
  updateScreen();
  keepClock();
  if (millis() - batteryAt > 60000) {
    batteryAt = millis();
    readBattery();
  }
  if (state == FAILED && millis() - failedAt > CARD_RETRY_MS) retryCard();
  while ((state == LISTENING || state == RECORDING) && head - scan >= FRAME) {
    bool sound = soundFrame();
    recent = (recent << 1) | sound;
    if (state == LISTENING) {
      if (scan - tail > PREROLL_BYTES) tail = scan - PREROLL_BYTES;  // forget older audio
      if (__builtin_popcount(recent) >= START_FRAMES) startSegment();
    } else if (sound) {
      lastSoundAt = scan;
      soundFrames++;
    } else if (scan - lastSoundAt > HANGOVER_BYTES) {
      endSegment();
    }
  }
  bool busy = false;
  // All the audio that's ready (writes never pass detection), so a stall elsewhere is made up.
  for (int i = 0; i < 16 && state == RECORDING && scan - tail >= BLOCK_PCM; i++) {
    if (!writeBlock(BLOCK_PCM)) fail();
    busy = true;
  }
  for (int i = 0; i < 8 && stepLive(); i++) busy = true;
  // Transfers get about 20 ms a turn: notify() waits whenever the radio's queue is full, and the
  // audio mustn't wait behind it for long.
  uint32_t start = millis();
  while (job != NO_JOB && millis() - start < 20 && stepJob()) busy = true;
  if (!busy) delay(10);  // lets the CPU sleep between blocks
}
