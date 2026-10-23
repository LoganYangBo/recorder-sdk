// The Bluefruit (Bluetooth LE) calls recorder.ino makes, with a radio that sim.cpp drains.
#pragma once
#include <deque>
#include <string>

#include "Arduino.h"

#define BLE_CONN_HANDLE_INVALID 0xFFFF
#define BLE_GAP_PHY_2MBPS 2
#define BLE_GATTC_WRITE_CMD_TX_QUEUE_SIZE_DEFAULT 1
#define BLE_GAP_ADV_FLAGS_LE_ONLY_GENERAL_DISC_MODE 6
#define CHR_PROPS_WRITE 8
#define CHR_PROPS_WRITE_WO_RESP 4
#define CHR_PROPS_NOTIFY 16
#define SECMODE_OPEN 1
#define SECMODE_NO_ACCESS 0
#define NRF_POWER_DCDC_ENABLE 1

struct ble_gap_conn_params_t {
  uint16_t min_conn_interval, max_conn_interval, slave_latency, conn_sup_timeout;
};

namespace sim {
struct Packet {
  std::string chr;
  std::string bytes;
};
extern std::deque<Packet> radio;  // notifications queued, not yet sent
extern uint8_t queueLimit;        // the largest queue the SoftDevice's RAM allows
extern uint16_t interval;         // connection interval, 1.25 ms units
extern bool dcdc;
bool waitForRoom();  // runs the clock until the queue has room (up to 100 ms)
}  // namespace sim

inline uint32_t sd_ble_gap_conn_param_update(uint16_t, const ble_gap_conn_params_t* p) {
  sim::interval = p->min_conn_interval;
  return 0;
}
inline uint32_t sd_power_dcdc_mode_set(uint8_t on) {
  sim::dcdc = on;
  return 0;
}
inline uint32_t sd_softdevice_disable() { return 0; }

class BLEConnection {
 public:
  uint16_t getMtu() { return 247; }
  uint16_t getConnectionInterval() { return sim::interval; }
  bool requestPHY(uint8_t) { return true; }
  bool requestDataLengthUpdate() { return true; }
  bool requestMtuExchange(uint16_t) { return true; }
  bool requestConnectionParameter(uint16_t i) {
    sim::interval = i;
    return true;
  }
};

class BLECharacteristic;
typedef void (*write_cb)(uint16_t, BLECharacteristic*, uint8_t*, uint16_t);

class BLECharacteristic {
 public:
  BLECharacteristic(const uint8_t* uuid) : name(uuid[12] == 2 ? "CTRL" : uuid[12] == 3 ? "DATA" : "LIVE") {}
  void setProperties(int) {}
  void setPermission(int, int) {}
  void setMaxLen(int) {}
  void setWriteCallback(write_cb cb) { onWrite = cb; }
  void begin() {}
  bool notify(const void* data, uint16_t len);
  const char* name;
  write_cb onWrite = nullptr;
};

class BLEService {
 public:
  BLEService(const uint8_t*) {}
  void begin() {}
};

class BLEBas {
 public:
  void begin() {}
  void write(int) {}
  void notify(int) {}
};

class SimBluefruit {
 public:
  void configPrphConn(uint16_t, uint16_t, uint8_t q, uint8_t) { queue = q; }
  bool begin() { return queue <= sim::queueLimit; }
  void autoConnLed(bool) {}
  void setName(const char*) {}
  bool connected() { return handle != BLE_CONN_HANDLE_INVALID; }
  BLEConnection* Connection(uint16_t h) { return h == handle ? &conn : nullptr; }
  void disconnect(uint16_t);
  void getAddr(uint8_t mac[6]) {
    const uint8_t a[6] = {0x5e, 0xc0, 0x11, 0x22, 0x33, 0xc4};
    memcpy(mac, a, 6);
  }
  struct {
    void (*onConnect)(uint16_t) = nullptr;
    void (*onDisconnect)(uint16_t, uint8_t) = nullptr;
    void setConnectCallback(void (*cb)(uint16_t)) { onConnect = cb; }
    void setDisconnectCallback(void (*cb)(uint16_t, uint8_t)) { onDisconnect = cb; }
  } Periph;
  struct {
    bool on = false;
    void addFlags(int) {}
    void addService(BLEService&) {}
    void restartOnDisconnect(bool) {}
    void setInterval(int, int) {}
    void setFastTimeout(int) {}
    void start(int) { on = true; }
    void stop() { on = false; }
  } Advertising;
  struct {
    void addName() {}
  } ScanResponse;
  uint8_t queue = 1;
  uint16_t handle = BLE_CONN_HANDLE_INVALID;
  BLEConnection conn;
};
extern SimBluefruit Bluefruit;
