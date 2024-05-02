#ifndef MATH_SPHERICAL_H
#define MATH_SPHERICAL_H
#include "math_utils_approx.h"
#include <glm/common.hpp>
#include <glm/glm.hpp>

constexpr double PI = M_PI;
namespace math::spherical {
  float acos_lt(float a);

  template<class T>
  inline glm::dvec2 uvToSpherical(const T &u, const T &v) {
    const T phi = 2 * PI * u;
    const T theta = PI * v;
    return glm::dvec2(phi, theta);
  }

  inline glm::dvec2 uvToSpherical(const glm::dvec2 &uv) { return uvToSpherical(uv.x, uv.y); }

  template<class T>
  inline glm::dvec2 sphericalToUv(const T &phi, const T &theta) {
    const T u = phi / (2 * PI);
    const T v = theta / PI;
    return {u, v};
  }

  inline glm::dvec2 sphericalToUv(const glm::dvec2 &sph) { return sphericalToUv(sph.x, sph.y); }

  template<class T>
  inline glm::dvec3 sphericalToCartesian(const T &phi, const T &theta) {
    const T z = cos(theta);
    const T x = sin(theta) * cos(phi);
    const T y = sin(theta) * sin(phi);
    return glm::dvec3(x, y, z);
  }

  inline glm::dvec3 sphericalToCartesian(const glm::dvec2 &sph) { return sphericalToCartesian(sph.x, sph.y); }

  template<class T>
  inline glm::dvec2 cartesianToSpherical(const T &x, const T &y, const T &z, bool fast_approx = true) {
    const T theta = fast_approx ? acos_lt(z) : std::acos(z);
    const T phi = fast_approx ? atan2_approx(y, x) : atan2f(y, x);
    return {phi, theta};
  }

  inline glm::dvec2 cartesianToSpherical(const glm::dvec3 &xyz, bool fast_approx = true) {
    return cartesianToSpherical(xyz.x, xyz.y, xyz.z, fast_approx);
  }

  struct SphAxis {
    float yaw;
    float pitch;
    float roll;
  };

  inline SphAxis cartesianToSphericalAxis(const glm::dvec3 &xyz) {
    return {atan2_approx(-xyz.x, -xyz.z), (float)std::asin(xyz.y / glm::length(xyz)), 0.f};
  }
}  // namespace math::spherical
#endif  // math_spherical_H
