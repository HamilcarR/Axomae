#include "math_spherical.h"
#include "math_utils_approx.h"
#include <bits/stl_algo.h>
#include <boost/math/ccmath/ccmath.hpp>
#include <internal/macro/project_macros.h>

extern "C" const float _binary_acos_table_start[];

constexpr float math::acos_approx(float x) {
  float negate = x < 0;
  x = boost::math::ccmath::abs(x);
  float ret = -0.0187293;
  ret = ret * x;
  ret = ret + 0.0742610;
  ret = ret * x;
  ret = ret - 0.2121144;
  ret = ret * x;
  ret = ret + 1.5707288;
  ret = ret * boost::math::ccmath::sqrt(1.0f - x);
  ret = ret - 2 * negate * ret;
  return static_cast<float>(negate * PI + ret);
}

namespace math::spherical {
  namespace acos {

    constexpr int lut_precision = PRECISION_LUT;

    inline float lut(int pos) {
      AX_ASSERT_LT(pos, lut_precision);
      const float *table = _binary_acos_table_start;
      return table[pos];
    }

  }  // namespace acos

  float acos_lt(float a) {
    int scaled = std::clamp((int)((a + 1.f) * acos::lut_precision * 0.5f), 0, acos::lut_precision - 1);
    return acos::lut(scaled);
  }
}  // namespace math::spherical
