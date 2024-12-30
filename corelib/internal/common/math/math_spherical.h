#ifndef MATH_SPHERICAL_H
#define MATH_SPHERICAL_H

#include "math_random_interface.h"
#include "math_utils_approx.h"

constexpr double PI = M_PI;
constexpr double INV_PI = 1.f / PI;
constexpr double INV_2PI = 1.f / (2.f * PI);

namespace math::spherical {
  float acos_lt(float a);

  template<class T>
  glm::dvec2 uvToSpherical(const T &u, const T &v) {
    const T phi = 2 * PI * u;
    const T theta = PI * v;
    return glm::dvec2(phi, theta);
  }

  inline glm::dvec2 uvToSpherical(const glm::dvec2 &uv) { return uvToSpherical(uv.x, uv.y); }

  template<class T>
  glm::dvec2 sphericalToUv(const T &phi, const T &theta) {
    const T u = phi / (2 * PI);
    const T v = theta / PI;
    return {u, v};
  }

  inline glm::dvec2 sphericalToUv(const glm::dvec2 &sph) { return sphericalToUv(sph.x, sph.y); }

  template<class T>
  glm::dvec3 sphericalToCartesian(const T &phi, const T &theta) {
    const T z = cos(theta);
    const T x = sin(theta) * cos(phi);
    const T y = sin(theta) * sin(phi);
    return glm::dvec3(x, y, z);
  }

  inline glm::dvec3 sphericalToCartesian(const glm::dvec2 &sph) { return sphericalToCartesian(sph.x, sph.y); }

  template<class T>
  glm::dvec2 cartesianToSpherical(const T &x, const T &y, const T &z, bool fast_approx = true) {
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

  template<class T>
  glm::vec3 rand_p_hemisphere(random::AbstractRandomGenerator<T> &generator) {
    float phi = generator.nrandf(0.f, 2 * PI);
    float theta = generator.nrandf(0.f, PI * 0.5f);
    return normalize(sphericalToCartesian(phi, theta));
  }

  template<class T>
  glm::vec3 rand_p_sphere(random::AbstractRandomGenerator<T> &generator) {
    glm::vec3 p;
    do {
      float x = generator.nrandf(0, 1);
      float y = generator.nrandf(0, 1);
      float z = generator.nrandf(0, 1);
      p = 2.f * glm::vec3(x, y, z) - glm::vec3(1, 1, 1);
    } while ((p.x * p.x + p.y * p.y + p.z * p.z) >= 1.f);
    return p;
  }

}  // namespace math::spherical
#endif  // math_spherical_H
