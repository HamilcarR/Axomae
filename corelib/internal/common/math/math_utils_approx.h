#ifndef MATH_UTILS_APPROX_H
#define MATH_UTILS_APPROX_H
#include <array>
#include <cmath>
#include <cstdlib>
#include <glm/common.hpp>
/**
 * trig functions optimization / approximations
 */

namespace math {
  inline float horner_approx(float x) {
    float a1 = 0.99997726f;
    float a3 = -0.33262347f;
    float a5 = 0.19354346f;
    float a7 = -0.11643287f;
    float a9 = 0.05265332f;
    float a11 = -0.01172120f;
    float x_sq = x * x;
    return x * (a1 + x_sq * (a3 + x_sq * (a5 + x_sq * (a7 + x_sq * (a9 + x_sq * a11)))));
  }

  inline float atan2_approx(const float ys, const float xs) {
    const float PI = M_PI;
    const float PI_2 = M_PI_2;
    float y = ys;
    float x = xs;
    bool swap = std::abs(x) < std::abs(y);
    float atan_input = (swap ? x : y) / (swap ? y : x);
    float res = horner_approx(atan_input);
    res = swap ? (atan_input >= 0.0f ? PI_2 : -PI_2) - res : res;
    if (x >= 0.0f && y >= 0.0f) {
    } else if (x < 0.0f && y >= 0.0f) {
      res = PI + res;
    } else if (x < 0.0f && y < 0.0f) {
      res = -PI + res;
    }
    return res;
  }

  inline float acos_approx(float x) {
    float negate = x < 0;
    x = std::abs(x);
    float ret = -0.0187293;
    ret = ret * x;
    ret = ret + 0.0742610;
    ret = ret * x;
    ret = ret - 0.2121144;
    ret = ret * x;
    ret = ret + 1.5707288;
    ret = ret * sqrt(1.0 - x);
    ret = ret - 2 * negate * ret;
    return negate * M_PI + ret;
  }

}  // namespace math

#endif  // MATH_UTILS_APPROX_H
