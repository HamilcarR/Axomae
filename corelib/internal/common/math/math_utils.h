#ifndef MATH_UTILS_H
#define MATH_UTILS_H

#include "math_importance_sampling.h"
#include "math_includes.h"
#include "math_spherical.h"
#include "math_texturing.h"

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

    constexpr bool is_nan_(const glm::vec3 &val) { return std::isnan(val.x) || std::isnan(val.y) || std::isnan(val.z); }

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
