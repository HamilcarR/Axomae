#include "Axomae_macros.h"
#include "math_spherical.h"

#include <bits/stl_algo.h>

namespace math::spherical {
  namespace acos {
    constexpr int lt_width = 200000;

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