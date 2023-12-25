#ifndef IMAGES_H
#define IMAGES_H
#include "constants.h"
#include <cstring>
namespace hdr_utils {
  inline void float2rgbe(unsigned char rgbe[4], float r, float g, float b) {
    float mantissa;
    int exponent;
    mantissa = std::max(std::max(r, g), b);
    if (mantissa < 1e-32)
      std::memset(rgbe, 0, 4);
    else {
      mantissa = std::frexp(mantissa, &exponent) * 256 / mantissa;
      rgbe[0] = (unsigned char)(r * mantissa);
      rgbe[1] = (unsigned char)(g * mantissa);
      rgbe[2] = (unsigned char)(b * mantissa);
      rgbe[3] = (unsigned char)(mantissa + 128);
    }
  }

  inline void rgbe2float(float rgbe, float rgb[3]) {
    float f;
    if (rgbe) {
      uint8_t exp = *((uint32_t *)&rgbe) << 24;
      uint8_t b = *((uint32_t *)&rgbe) << 16;
      uint8_t g = *((uint32_t *)&rgbe) << 8;
      uint8_t r = *((uint32_t *)&rgbe) & 0xFF;
      f = std::ldexp(1, exp - (int)(128 + 8));
      rgb[0] = r * f;
      rgb[1] = g * f;
      rgb[2] = b * f;
    }
  }

};  // namespace hdr_utils

#endif
