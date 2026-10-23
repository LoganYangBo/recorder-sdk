// The part of SdFat that recorder.ino uses, over a folder on the computer (the "card").
// It also notes every write, so tests can check they end on sector boundaries.
#pragma once
#include <fcntl.h>

#include <algorithm>
#include <filesystem>
#include <string>
#include <vector>

#include "Arduino.h"

namespace sim {
extern std::filesystem::path card;  // the card's root folder
extern bool cardBroken;             // every card operation fails (the card was pulled out)
extern uint32_t writes, unalignedWrites;
}  // namespace sim

#define DEDICATED_SPI 1
#define SD_SCK_MHZ(m) ((m) * 1000000UL)
#define FS_DATE(y, m, d) ((uint16_t)(((y) - 1980) << 9 | (m) << 5 | (d)))
#define FS_TIME(h, m, s) ((uint16_t)((h) << 11 | (m) << 5 | (s) >> 1))
typedef int oflag_t;

struct SdSpiConfig {
  SdSpiConfig(int, int, uint32_t) {}
};

struct FsDateTime {
  static void setCallback(void (*)(uint16_t*, uint16_t*)) {}
};

class File32 {
 public:
  File32() {}
  File32(const File32& o) { *this = o; }
  File32& operator=(const File32& o) {
    if (this == &o) return *this;
    close();
    // like SdFat's copy: both refer to the same open file; the copy is what gets used
    path_ = o.path_, dir_ = o.dir_, entries_ = o.entries_, next_ = o.next_, pos_ = o.pos_,
    size_ = o.size_, open_ = o.open_, f_ = o.f_;
    const_cast<File32&>(o).open_ = false;
    const_cast<File32&>(o).f_ = nullptr;
    return *this;
  }
  ~File32() { close(); }

  bool open(const char* p, oflag_t flags = O_RDONLY) {
    close();
    if (sim::cardBroken) return false;
    path_ = sim::card / p;
    std::error_code ec;
    if (std::filesystem::is_directory(path_, ec)) {
      dir_ = open_ = true;
      for (auto& e : std::filesystem::directory_iterator(path_)) entries_.push_back(e.path());
      std::sort(entries_.begin(), entries_.end());
      return true;
    }
    bool exists = std::filesystem::exists(path_, ec);
    if (!exists && !(flags & O_CREAT)) return false;
    const char* mode = (flags & O_TRUNC) || !exists ? ((flags & O_RDWR) ? "w+b" : "wb")
                       : (flags & (O_RDWR | O_WRONLY)) ? "r+b"
                                                       : "rb";
    f_ = fopen(path_.string().c_str(), mode);
    if (!f_) return false;
    fseek(f_, 0, SEEK_END);
    size_ = ftell(f_);
    fseek(f_, 0, SEEK_SET);
    pos_ = 0;
    dir_ = false;
    return open_ = true;
  }
  bool close() {
    if (f_) fclose(f_);
    f_ = nullptr;
    bool was = open_;
    open_ = false;
    entries_.clear();
    next_ = 0;
    return was && !sim::cardBroken;
  }
  explicit operator bool() const { return open_; }
  bool isDir() const { return open_ && dir_; }
  bool isHidden() const { return false; }
  uint32_t fileSize() const { return size_; }
  uint32_t curPosition() const { return pos_; }
  size_t getName(char* name, size_t size) {
    snprintf(name, size, "%s", path_.filename().string().c_str());
    return strlen(name);
  }
  File32 openNextFile() {
    File32 f;
    while (next_ < entries_.size()) {
      auto rel = std::filesystem::relative(entries_[next_++], sim::card).generic_string();
      if (f.open(rel.c_str())) break;
    }
    return f;
  }
  int read(void* buf, size_t n) {
    if (sim::cardBroken || !f_) return -1;
    size_t got = fread(buf, 1, n, f_);
    pos_ += got;
    return (int)got;
  }
  size_t write(const void* buf, size_t n) {
    if (sim::cardBroken || !f_) return 0;
    size_t put = fwrite(buf, 1, n, f_);
    if (pos_ > 0) {  // the header, at 0, is rewritten in place: not counted
      sim::writes++;
      if ((pos_ + put) % 512) sim::unalignedWrites++;
    }
    pos_ += put;
    size_ = std::max(size_, pos_);
    return put;
  }
  bool seekSet(uint32_t p) {
    if (sim::cardBroken || !f_ || p > size_) return false;
    pos_ = p;
    return fseek(f_, p, SEEK_SET) == 0;
  }
  bool sync() { return !sim::cardBroken && f_ && fflush(f_) == 0; }

 private:
  std::filesystem::path path_;
  bool dir_ = false, open_ = false;
  std::vector<std::filesystem::path> entries_;
  size_t next_ = 0;
  uint32_t pos_ = 0, size_ = 0;
  FILE* f_ = nullptr;
};

class SdFat32 {
 public:
  bool begin(SdSpiConfig) { return !sim::cardBroken; }
  void end() {}
  bool exists(const char* p) {
    std::error_code ec;
    return !sim::cardBroken && std::filesystem::exists(sim::card / p, ec);
  }
  bool mkdir(const char* p) {
    std::error_code ec;
    if (sim::cardBroken) return false;
    std::filesystem::create_directories(sim::card / p, ec);
    return !ec;
  }
  bool remove(const char* p) {
    std::error_code ec;
    auto full = sim::card / p;
    if (sim::cardBroken || !std::filesystem::is_regular_file(full, ec)) return false;
    return std::filesystem::remove(full, ec);
  }
  bool rmdir(const char* p) {
    std::error_code ec;
    auto full = sim::card / p;
    if (sim::cardBroken || !std::filesystem::is_directory(full, ec) ||
        !std::filesystem::is_empty(full, ec))
      return false;
    return std::filesystem::remove(full, ec);
  }
  File32 open(const char* p, oflag_t flags = O_RDONLY) {
    File32 f;
    f.open(p, flags);
    return f;
  }
};
