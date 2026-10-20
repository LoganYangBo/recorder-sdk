// IMA ADPCM, the 4:1 compression of WAV format 0x11 that ffmpeg/PyAV decode.
// A block starts with the first sample and the step index (4 bytes), then two samples per byte,
// low nibble first: a block of B bytes holds 1 + (B - 4) * 2 samples and decodes on its own.
// The decoder side is src/recorder/adpcm.py; tests/test_adpcm.py compiles this file to check
// that the two agree.
#pragma once
#include <stdint.h>

static const int16_t ADPCM_STEPS[89] = {
    7,     8,     9,     10,    11,    12,    13,    14,    16,    17,    19,    21,    23,
    25,    28,    31,    34,    37,    41,    45,    50,    55,    60,    66,    73,    80,
    88,    97,    107,   118,   130,   143,   157,   173,   190,   209,   230,   253,   279,
    307,   337,   371,   408,   449,   494,   544,   598,   658,   724,   796,   876,   963,
    1060,  1166,  1282,  1411,  1552,  1707,  1878,  2066,  2272,  2499,  2749,  3024,  3327,
    3660,  4026,  4428,  4871,  5358,  5894,  6484,  7132,  7845,  8630,  9493,  10442, 11487,
    12635, 13899, 15289, 16818, 18500, 20350, 22385, 24623, 27086, 29794, 32767};
static const int8_t ADPCM_INDEX[16] = {-1, -1, -1, -1, 2, 4, 6, 8, -1, -1, -1, -1, 2, 4, 6, 8};

struct Adpcm {
  int32_t pred = 0;
  int32_t index = 0;
};

static inline uint8_t adpcmNibble(Adpcm& s, int16_t x) {
  int32_t step = ADPCM_STEPS[s.index];
  int32_t diff = x - s.pred;
  uint8_t code = 0;
  if (diff < 0) {
    code = 8;
    diff = -diff;
  }
  int32_t delta = step >> 3;
  if (diff >= step) {
    code |= 4;
    diff -= step;
    delta += step;
  }
  if (diff >= step >> 1) {
    code |= 2;
    diff -= step >> 1;
    delta += step >> 1;
  }
  if (diff >= step >> 2) {
    code |= 1;
    delta += step >> 2;
  }
  s.pred += (code & 8) ? -delta : delta;
  if (s.pred > 32767) s.pred = 32767;
  if (s.pred < -32768) s.pred = -32768;
  s.index += ADPCM_INDEX[code];
  if (s.index < 0) s.index = 0;
  if (s.index > 88) s.index = 88;
  return code;
}

// Encodes 1 + (bytes - 4) * 2 samples from `x` into a block of `bytes` bytes. `tag` goes in the
// header's spare byte (0 in files; the live stream puts a packet counter there).
static inline void adpcmBlock(Adpcm& s, const int16_t* x, uint8_t* out, int bytes, uint8_t tag) {
  s.pred = x[0];
  out[0] = (uint16_t)x[0] & 0xFF;
  out[1] = (uint16_t)x[0] >> 8;
  out[2] = (uint8_t)s.index;
  out[3] = tag;
  for (int i = 4, k = 1; i < bytes; i++, k += 2) {
    uint8_t lo = adpcmNibble(s, x[k]);
    out[i] = lo | (adpcmNibble(s, x[k + 1]) << 4);
  }
}
