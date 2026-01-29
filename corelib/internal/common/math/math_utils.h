#ifndef MATH_UTILS_H
#define MATH_UTILS_H

#include "math_importance_sampling.h"
#include "math_includes.h"
#include "math_spherical.h"
#include "math_texturing.h"

#define AX_ASSERT_NOTNAN(val) assert(!ISNAN(val) && !ISINF(val));
struct Dim2 {
  unsigned width;
  unsigned height;
};
namespace math {

  namespace internals {
    template<typename T>
    constexpr bool is_nan_(const T &val) {
      return std::isnan(val);
    }

    template<>
    constexpr bool is_nan_(const glm::vec3 &val) {
      return std::isnan(val.x) || std::isnan(val.y) || std::isnan(val.z);
    }

    template<>
    constexpr bool is_nan_(const glm::vec4 &val) {
      return std::isnan(val.x) || std::isnan(val.y) || std::isnan(val.z) || std::isnan(val.w);
    }

    template<>
    constexpr bool is_nan_(const glm::vec2 &val) {
      return std::isnan(val.x) || std::isnan(val.y);
    }

    template<typename T>
    constexpr bool is_inf_(const T &val) {
      return std::isinf(val);
    }

    template<>
    constexpr bool is_inf_(const glm::vec3 &val) {
      return std::isinf(val.x) || std::isinf(val.y) || std::isinf(val.z);
    }

    template<>
    constexpr bool is_inf_(const glm::vec4 &val) {
      return std::isinf(val.x) || std::isinf(val.y) || std::isinf(val.z) || std::isinf(val.w);
    }

    template<>
    constexpr bool is_inf_(const glm::vec2 &val) {
      return std::isinf(val.x) || std::isinf(val.y);
    }

    template<typename T>
    constexpr T denan_(T val) {
      return {};
    }

  }  // namespace internals

  inline constexpr double epsilon = 1e-6;
  namespace calculus {

    /* Will compute X in :
     * N = 1 + 2 + 3 + 4 + 5 + ... + X
     */
    template<class T>
    float compute_serie_term(T N) {
      return (-1 + std::sqrt(1.f + 8 * N)) * 0.5f;
    }

  }  // namespace calculus

  template<class T>
  ax_device_callable_inlined constexpr T sqr(const T &val) {
    return val * val;
  }

  template<class T>
  ax_device_callable_inlined constexpr T sqrt(T val);

  template<>
  ax_device_callable_inlined constexpr float sqrt(float val) {
    return sqrtf(fmax(val, 0.f));
  }

  template<>
  ax_device_callable_inlined constexpr double sqrt(double val) {
    return sqrt(fmax(val, 0.));
  }

  template<class T>
  ax_device_callable_inlined constexpr T lerp(const T &x, const T &y, const T &a) {
    return x * (static_cast<T>(1) - a) + a * y;
  }

  ax_device_callable_inlined uint32_t tea_hash(uint32_t v0, uint32_t v1) {
    uint32_t sum = 0;
    for (int i = 0; i < 4; i++) {
      sum += 0x9e3779b9;
      v0 += ((v1 << 4) + 0xa341316c) ^ (v1 + sum) ^ ((v1 >> 5) + 0xc8013ea4);
      v1 += ((v0 << 4) + 0xad90777d) ^ (v0 + sum) ^ ((v0 >> 5) + 0x7e95761e);
    }
    return v0;
  }

}  // namespace math

template<class T>
constexpr bool ISNAN(const T &val) {
  return math::internals::is_nan_(val);
}

template<class T>
constexpr bool ISINF(const T &val) {
  return math::internals::is_inf_(val);
}

template<class T>
constexpr T DENAN(const T &val) {
  return (ISINF(val) || ISNAN(val)) ? math::internals::denan_(val) : val;
}

template<class T>
constexpr T DENAN(const T &val, const T &default_) {
  return (ISINF(val) || ISNAN(val)) ? default_ : val;
}

#endif
