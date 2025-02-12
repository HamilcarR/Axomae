#ifndef MATH_INCLUDES_H
#define MATH_INCLUDES_H
#define GLM_ENABLE_EXPERIMENTAL
#include <float.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/quaternion.hpp>
#include <math.h>
namespace math::internals {
  template<typename T>
  constexpr bool is_nan_(T val) {
    return std::isnan(val);
  }

  constexpr bool is_nan_(const glm::vec3 val) { return std::isnan(val.x) || std::isnan(val.y) || std::isnan(val.z); }

  template<>
  constexpr bool is_nan_(const glm::vec4 val) {
    return std::isnan(val.x) || std::isnan(val.y) || std::isnan(val.z) || std::isnan(val.w);
  }

  template<>
  constexpr bool is_nan_(const glm::vec2 val) {
    return std::isnan(val.x) || std::isnan(val.y);
  }

  template<typename T>
  constexpr bool is_inf_(T val) {
    return std::isinf(val);
  }

  template<>
  constexpr bool is_inf_(const glm::vec3 val) {
    return std::isinf(val.x) || std::isinf(val.y) || std::isinf(val.z);
  }

  template<>
  constexpr bool is_inf_(const glm::vec4 val) {
    return std::isinf(val.x) || std::isinf(val.y) || std::isinf(val.z) || std::isinf(val.w);
  }

  template<>
  constexpr bool is_inf_(const glm::vec2 val) {
    return std::isinf(val.x) || std::isinf(val.y);
  }

  template<typename T>
  constexpr T denan_(T val) {
    return {};
  }

}  // namespace math::internals

template<class T>
constexpr bool ISNAN(T val) {
  return math::internals::is_nan_(val);
}

template<class T>
constexpr bool ISINF(T val) {
  return math::internals::is_inf_(val);
}

template<class T>
constexpr T DENAN(T val) {
  return (ISINF(val) || ISNAN(val)) ? math::internals::denan_(val) : val;
}

#endif  // MATH_INCLUDES_H
