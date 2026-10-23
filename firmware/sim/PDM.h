// The PDM microphone: sim.cpp calls the callback every 16 ms with samples from a file.
#pragma once
#include "Arduino.h"

class PDMClass {
 public:
  void onReceive(void (*cb)()) { callback = cb; }
  int begin(int, long) { return running = true; }
  void end() { running = false; }
  void setGain(int g) { gain = g; }
  int available() { return pending; }
  int read(void* buf, int n) {
    n = min(n, pending);
    memcpy(buf, data, n);
    pending = 0;
    return n;
  }
  void (*callback)() = nullptr;
  bool running = false;
  int gain = 20, pending = 0;
  int16_t data[256];
};
extern PDMClass PDM;
