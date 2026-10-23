// Just enough of the Arduino core for recorder.ino to run on a computer (see sim.cpp).
#pragma once
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <math.h>
#include <time.h>

#include <string>
#include <type_traits>

typedef uint8_t byte;

#define HIGH 1
#define LOW 0
#define INPUT 0
#define OUTPUT 1
#define INPUT_PULLUP 2
#define LED_BUILTIN 11
#define LED_BLUE 12
#define PIN_LSM6DS3TR_C_POWER 15
#define AR_INTERNAL_3_0 0

template <class A, class B>
auto min(A a, B b) -> typename std::decay<decltype(a < b ? a : b)>::type {
  return a < b ? a : b;
}
template <class A, class B>
auto max(A a, B b) -> typename std::decay<decltype(a < b ? a : b)>::type {
  return a < b ? b : a;
}

namespace sim {
uint32_t millis();
void advance(uint32_t ms);
int digitalRead(int pin);
void digitalWrite(int pin, int value);
uint32_t resetReason();
void serialOut(char c);
int serialAvailable();
int serialRead();
}  // namespace sim

inline uint32_t millis() { return sim::millis(); }
inline void delay(uint32_t ms) { sim::advance(ms); }
inline void yield() {}
inline void pinMode(int, int) {}
inline int digitalRead(int pin) { return sim::digitalRead(pin); }
inline void digitalWrite(int pin, int value) { sim::digitalWrite(pin, value); }
inline int analogRead(int) { return 1798; }  // 3.90 V through the divider
inline void analogReference(int) {}
inline void analogReadResolution(int) {}
inline uint32_t readResetReason() { return sim::resetReason(); }

class Print {
 public:
  virtual ~Print() {}
  virtual size_t write(uint8_t c) = 0;
  virtual size_t write(const uint8_t* buf, size_t n) {
    for (size_t i = 0; i < n; i++) write(buf[i]);
    return n;
  }
  size_t print(const char* s) { return write((const uint8_t*)s, strlen(s)); }
  size_t print(char c) { return write((uint8_t)c); }
  size_t print(int v) { return printf_("%d", v); }
  size_t print(unsigned v) { return printf_("%u", v); }
  size_t print(long v) { return printf_("%ld", v); }
  size_t print(unsigned long v) { return printf_("%lu", v); }
  size_t print(double v, int digits = 2) { return printf_("%.*f", digits, v); }
  size_t println() { return write((uint8_t)'\n'); }
  template <class T>
  size_t println(T v) {
    size_t n = print(v);
    return n + println();
  }
  template <class T>
  size_t println(T v, int digits) {
    size_t n = print(v, digits);
    return n + println();
  }

 private:
  template <class... A>
  size_t printf_(const char* fmt, A... a) {
    char buf[32];
    int n = snprintf(buf, sizeof buf, fmt, a...);
    return write((const uint8_t*)buf, n);
  }
};

class SimSerial : public Print {
 public:
  void begin(long) {}
  int available() { return sim::serialAvailable(); }
  int read() { return sim::serialRead(); }
  size_t write(uint8_t c) override {
    sim::serialOut(c);
    return 1;
  }
  using Print::write;
};
extern SimSerial Serial;

// The nRF52's watchdog and reset reasons, as registers.
struct NRF_WDT_Type {
  uint32_t RUNSTATUS, CONFIG, CRV, RREN, TASKS_START, RR[8];
};
extern NRF_WDT_Type sim_wdt;
#define NRF_WDT (&sim_wdt)
#define WDT_CONFIG_SLEEP_Run 1UL
#define WDT_CONFIG_SLEEP_Pos 0
#define WDT_CONFIG_HALT_Pause 0UL
#define WDT_CONFIG_HALT_Pos 3
#define WDT_RREN_RR0_Msk 1UL
#define WDT_RR_RR_Reload 0x6E524635UL
#define POWER_RESETREAS_RESETPIN_Msk (1UL << 0)
#define POWER_RESETREAS_DOG_Msk (1UL << 1)
#define POWER_RESETREAS_SREQ_Msk (1UL << 2)
#define POWER_RESETREAS_LOCKUP_Msk (1UL << 3)
