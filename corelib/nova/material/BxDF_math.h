#ifndef BXDF_MATH_H
#define BXDF_MATH_H
#include <cmath>
#include <internal/common/math/math_complex.h>
#include <internal/common/math/math_spherical.h>
#include <internal/common/math/math_utils.h>
#include <internal/device/gpgpu/device_macros.h>
#include <internal/device/gpgpu/device_utils.h>
#include <math.h>

/* Vectors are assumed to be in tangent space. */
namespace bxdf {

  ax_device_callable_inlined float costheta(const glm::vec3 &vector) { return vector.z; }
  ax_device_callable_inlined float abscostheta(const glm::vec3 &direction) { return fabsf(costheta(direction)); }

  /* Provides a uniform mapping between square to disk .*/
  ax_device_callable_inlined glm::vec2 shirley_chiu(const float u[2]) {
    float sx = 2 * u[0] - 1;
    float sy = 2 * u[1] - 1;

    if (sx == 0 && sy == 0)
      return glm::vec2(0.f);

    float r = 0.f;
    float theta = 0.f;
    if (fabsf(sx) > fabsf(sy)) {
      r = sx;
      theta = M_PI_4f * (sy / sx);
    } else {
      r = sy;
      theta = M_PI_2f - M_PI_4f * (sx / sy);
    }
    return r * glm::vec2(cos(theta), sin(theta));
  }

  ax_device_callable_inlined glm::vec3 hemisphere_cosine_sample(const float u[2]) {
    glm::vec2 sph = shirley_chiu(u);
    float z = sqrtf(std::max(0.f, 1 - sph.x * sph.x - sph.y * sph.y));
    return {sph.x, sph.y, z};
  }

  ax_device_callable_inlined glm::vec3 hemisphere_sample_uniform(const float u[2]) {
    glm::vec2 sph = math::spherical::uvToSpherical(u[0], u[1]);
    return math::spherical::sphericalToCartesian(sph);
  }

  ax_device_callable_inlined constexpr float hemisphere_pdf() { return INV_2PI; }

}  // namespace bxdf

class Fresnel {

 public:
  // Dielectrics.
  ax_device_callable_inlined float Frf(float costheta_i, float eta) {
    costheta_i = glm::clamp(costheta_i, -1.f, 1.f);
    if (costheta_i < 0) {
      eta = 1.f / eta;
      costheta_i = -costheta_i;
    }
    float sin2theta_i = 1 - math::sqr(costheta_i);
    float sin2theta_t = sin2theta_i / math::sqr(eta);
    if (sin2theta_t >= 1)
      return 1.f;
    float costheta_t = math::sqrt(1.f - sin2theta_t);
    float rpar = (eta * costheta_i - costheta_t) / (eta * costheta_i + costheta_t);
    float rperp = (costheta_i - eta * costheta_t) / (costheta_i + eta * costheta_t);
    return (math::sqr(rpar) + math::sqr(rperp)) / 2.f;
  }

  // Conductors
  ax_device_callable_inlined float Frc(float costheta_i, const math::fcomplex &eta) {
    using fcomplex = math::fcomplex;
    costheta_i = glm::clamp(costheta_i, 0.f, 1.f);
    float sin2theta_i = 1 - math::sqr(costheta_i);
    fcomplex sin2theta_t = sin2theta_i / math::sqr(eta);
    fcomplex costheta_t = math::sqrt(1.f - sin2theta_t);
    if (costheta_t.imaginary < 0)
      costheta_t.imaginary *= -1.f;

    fcomplex R_parallel = (costheta_i * eta - costheta_t) / (costheta_i * eta + costheta_t);
    fcomplex R_perpendicular = (costheta_i - eta * costheta_t) / (costheta_i + eta * costheta_t);

    return (R_parallel.norm() + R_perpendicular.norm()) / 2.f;
  }
};
/* Vectors in tangent space.*/
class GGX {
  float alpha_x, alpha_x_inv;
  float alpha_y, alpha_y_inv;

  ax_device_callable_inlined float roughnessToAlpha(float roughness) const { return exp2(roughness * 10.f - 10.f); }
  ax_device_callable_inlined float lambda(const glm::vec3 &v) const {
    float alpha_x_term = math::sqr(alpha_x * v.x);
    float alpha_y_term = math::sqr(alpha_y * v.y);
    float z_term = math::sqr(v.z);
    float total_term = (alpha_x_term + alpha_y_term) / z_term;
    AX_ASSERT_GE(total_term, 0.f);
    float sqrt_term = sqrtf(1 + total_term);
    return (-1 + sqrt_term) * 0.5f;
  }

 public:
  ax_device_callable_inlined GGX(float roughness) {
    alpha_x = roughnessToAlpha(roughness);
    alpha_y = alpha_x;

    alpha_x_inv = 1.f / alpha_x;
    alpha_y_inv = 1.f / alpha_y;
  }

  ax_device_callable_inlined GGX(float roughness_x, float roughness_y) {
    alpha_x = roughnessToAlpha(roughness_x);
    alpha_y = roughnessToAlpha(roughness_y);
    alpha_x_inv = 1.f / alpha_x;
    alpha_y_inv = 1.f / alpha_y;
  }

  ax_device_callable_inlined float D(glm::vec3 h) const {
    if (h.z == 0.f)
      return 0.f;
    float x_ratio = math::sqr(h.x) * math::sqr(alpha_x_inv);
    float y_ratio = math::sqr(h.y) * math::sqr(alpha_y_inv);
    float z_ratio = math::sqr(h.z);
    float squared_denominator = math::sqr(x_ratio + y_ratio + z_ratio);
    return alpha_x_inv * alpha_y_inv * M_1_PIf * 1.f / squared_denominator;
  }

  ax_device_callable_inlined float G1(glm::vec3 v) const {
    v.z = std::max(v.z, 1e-4f);
    return 1.f / (1.f + lambda(v));
  }

  /* Takes a normal(half-vector) and a view vector.*/
  ax_device_callable_inlined float Dv(const glm::vec3 &h, const glm::vec3 &v) const {
    AX_ASSERT_GE(v.z, 0.f);
    float g1 = G1(v);
    float hdotv = std::max(0.f, glm::dot(h, v));
    float d = D(h);
    return g1 * hdotv * d / bxdf::abscostheta(v);
  }

  /* Checks if roughness allows for a specular lobe.*/
  ax_device_callable_inlined bool isFullSpecular() const { return std::max(alpha_x, alpha_y) < 1e-3; }
};

#endif
