#include "Axomae_macros.h"
#include "math_spherical.h"

#include <bits/stl_algo.h>

namespace math::spherical {
  namespace acos {
#ifdef LUT_HIGHP
    constexpr int lt_width = 1000000;
#elif LUT_MEDIUMP
    constexpr int lt_width = 500000;
#else
    constexpr int lt_width = 10000;
#endif
    constexpr float fill_lut(float i) { return std::acos(i); }

    constexpr auto LUT = [] {
      std::array<float, lt_width> table{};
      for (int i = 0; i < lt_width; i++) {
        float scaled = ((float)i / (float)lt_width) * 2.f - 1.f;
        table[i] = fill_lut(scaled);
      }
      return table;
    }();
  }  // namespace acos
  float acos_lt(float a) {
    int scaled = std::clamp((int)((a + 1.f) * acos::lt_width * 0.5f), 0, acos::lt_width - 1);
    return acos::LUT[scaled];
  }
}  // namespace math::spherical