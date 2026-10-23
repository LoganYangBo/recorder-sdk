// Runs recorder.ino on a computer, for tests (tests/test_firmware.py): the microphone plays a
// file, the card is a folder, and Bluetooth is a radio that sends a few queued notifications
// per connection interval. Time is simulated: it moves 1 ms per loop() and by every delay().
//
//   sim [--card DIR] [--reset power|button|watchdog|soft,...] [--kept EPOCH] [--queue-limit N]
//
// Commands on stdin, one per line; each answer ends with ". <millis>":
//   run MS              loop for MS ms of simulated time
//   audio PATH          the microphone plays PATH (16 kHz mono int16), then near-silence
//   connect / disconnect
//   write LINE          the computer writes LINE to CTRL; runs until the reply is all sent
//   writefor MS LINE    the same, but stops after MS ms whatever happened
//   serial LINE         LINE on the USB serial port
//   press MS            holds the button for MS ms
//   card broken|ok      every card operation fails / works again
//   stats               STATS writes=.. unaligned=.. dropped=.. barks=.. gain=.. dcdc=..
// Output: "N CTRL|DATA|LIVE <hex>" for each notification sent, "S <line>" for serial output.

#include <stdio.h>

#include <filesystem>
#include <iostream>
#include <string>

#include "Arduino.h"
#include "PDM.h"
#include "SdFat.h"
#include "bluefruit.h"

namespace sim {
std::filesystem::path card = "card";
bool cardBroken = false;
uint32_t writes = 0, unalignedWrites = 0;
std::deque<Packet> radio;
uint8_t queueLimit = 255;
uint16_t interval = 24;
bool dcdc = false;
uint64_t nowUs = 0, nextPdmUs = 16000, nextEventUs = 0, fedUs = 0;
uint32_t barks = 0, reset = 0;
int button = HIGH;
FILE* audio = nullptr;
uint32_t noise = 1;
std::string serialIn, serialLine;
const int PER_EVENT = 6;  // packets the computer takes per connection event

uint32_t millis() { return (uint32_t)(nowUs / 1000); }
int digitalRead(int pin) { return pin == 1 ? button : HIGH; }
void digitalWrite(int, int) {}
uint32_t resetReason() { return reset; }
void serialOut(char c) {
  if (c == '\n') {
    printf("S %s\n", serialLine.c_str());
    serialLine.clear();
  } else if (c != '\r') {
    serialLine += c;
  }
}
int serialAvailable() { return (int)serialIn.size(); }
int serialRead() {
  if (serialIn.empty()) return -1;
  int c = (unsigned char)serialIn[0];
  serialIn.erase(0, 1);
  return c;
}

void hex(const std::string& b) {
  static const char* d = "0123456789abcdef";
  for (unsigned char c : b) putchar(d[c >> 4]), putchar(d[c & 15]);
}

void micFrame() {
  size_t got = audio ? fread(PDM.data, 2, 256, audio) : 0;
  for (size_t i = got; i < 256; i++) {  // after the file: a quiet room
    noise = noise * 1103515245 + 12345;
    PDM.data[i] = (int16_t)((noise >> 16) % 5) - 2;
  }
  PDM.pending = 512;
  if (PDM.running && PDM.callback) PDM.callback();
}

void advance(uint32_t ms) {
  for (uint32_t i = 0; i < ms; i++) {
    nowUs += 1000;
    if (sim_wdt.RR[0] == WDT_RR_RR_Reload) fedUs = nowUs, sim_wdt.RR[0] = 0;
    if (sim_wdt.TASKS_START && nowUs - fedUs > (uint64_t)(sim_wdt.CRV + 1) * 1000000 / 32768) {
      barks++;  // the real one would reset here
      fedUs = nowUs;
    }
    while (nextPdmUs <= nowUs) {
      micFrame();
      nextPdmUs += 16000;
    }
    if (Bluefruit.connected()) {
      while (nextEventUs <= nowUs) {
        for (int k = 0; k < PER_EVENT && !radio.empty(); k++) {
          printf("N %s ", radio.front().chr.c_str());
          hex(radio.front().bytes);
          putchar('\n');
          radio.pop_front();
        }
        nextEventUs += interval * 1250;
      }
    }
  }
}

bool waitForRoom() {
  for (int i = 0; i < 100 && radio.size() >= Bluefruit.queue; i++) advance(1);
  return radio.size() < Bluefruit.queue;
}
}  // namespace sim

SimSerial Serial;
PDMClass PDM;
SimBluefruit Bluefruit;
NRF_WDT_Type sim_wdt;

bool BLECharacteristic::notify(const void* data, uint16_t len) {
  if (!Bluefruit.connected() || !sim::waitForRoom()) return false;
  sim::radio.push_back({name, std::string((const char*)data, len)});
  return true;
}

void SimBluefruit::disconnect(uint16_t) {
  if (!connected()) return;
  handle = BLE_CONN_HANDLE_INVALID;
  sim::radio.clear();
  if (Periph.onDisconnect) Periph.onDisconnect(1, 0x13);
}

#include "../recorder/recorder.ino"

namespace sim {
void runFor(uint32_t ms) {
  uint32_t end = millis() + ms;
  while ((int32_t)(end - millis()) > 0) {
    loop();
    advance(1);
  }
}

// Until the command has been run and its replies and data are all sent (or `limit` ms).
void runUntilIdle(uint32_t limit) {
  uint32_t end = millis() + limit;
  while ((int32_t)(end - millis()) > 0) {
    loop();
    advance(1);
    if (bleDone == bleIn && job == NO_JOB && pktLen == 0 && radio.empty()) {
      runFor(20);
      return;
    }
  }
}

void bleWrite(const std::string& line) {
  ctrlChr.onWrite(1, &ctrlChr, (uint8_t*)line.data(), (uint16_t)line.size());
}
}  // namespace sim

int main(int argc, char** argv) {
  using namespace sim;
  for (int i = 1; i + 1 < argc; i += 2) {
    std::string a = argv[i], v = argv[i + 1];
    if (a == "--card") card = v;
    if (a == "--queue-limit") queueLimit = (uint8_t)atoi(v.c_str());
    if (a == "--kept") {
      kept.time = (uint32_t)strtoul(v.c_str(), nullptr, 10);
      kept.check = kept.time ^ KEPT_MAGIC;
      kept.magic = KEPT_MAGIC;
    }
    if (a == "--reset") {
      if (v.find("button") != std::string::npos) reset |= POWER_RESETREAS_RESETPIN_Msk;
      if (v.find("watchdog") != std::string::npos) reset |= POWER_RESETREAS_DOG_Msk;
      if (v.find("soft") != std::string::npos) reset |= POWER_RESETREAS_SREQ_Msk;
    }
  }
  std::filesystem::create_directories(card);
  setup();
  std::string line;
  while (std::getline(std::cin, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    std::string cmd = line.substr(0, line.find(' '));
    std::string arg = line.find(' ') == std::string::npos ? "" : line.substr(line.find(' ') + 1);
    if (cmd == "run") {
      runFor((uint32_t)atol(arg.c_str()));
    } else if (cmd == "audio") {
      if (audio) fclose(audio);
      audio = fopen(arg.c_str(), "rb");
    } else if (cmd == "connect") {
      Bluefruit.handle = 1;
      nextEventUs = nowUs;
      if (Bluefruit.Periph.onConnect) Bluefruit.Periph.onConnect(1);
      runFor(10);
    } else if (cmd == "disconnect") {
      Bluefruit.disconnect(1);
      runFor(10);
    } else if (cmd == "write") {
      bleWrite(arg);
      runUntilIdle(3600 * 1000);
    } else if (cmd == "writefor") {
      std::string rest = arg.substr(arg.find(' ') + 1);
      bleWrite(rest);
      runFor((uint32_t)atol(arg.c_str()));
    } else if (cmd == "serial") {
      serialIn += arg + "\n";
      runFor(20);
    } else if (cmd == "press") {
      button = LOW;
      runFor((uint32_t)atol(arg.c_str()));
      button = HIGH;
      runFor(100);
    } else if (cmd == "card") {
      cardBroken = arg == "broken";
    } else if (cmd == "stats") {
      printf("STATS writes=%u unaligned=%u dropped=%u barks=%u gain=%d dcdc=%d state=%d\n",
             writes, unalignedWrites, (unsigned)dropped, barks, PDM.gain, dcdc, (int)state);
    }
    printf(". %u\n", sim::millis());
    fflush(stdout);
  }
  return 0;
}
