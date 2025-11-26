#ifndef BXDF_MATH_H
#define BXDF_MATH_H
#include <algorithm>
#include <cmath>
#include <internal/common/math/math_complex.h>
#include <internal/common/math/math_spherical.h>
#include <internal/common/math/math_utils.h>
#include <internal/common/math/utils_3D.h>
#include <internal/device/gpgpu/device_macros.h>
#include <internal/device/gpgpu/device_utils.h>
#include <internal/macro/project_macros.h>
#include <math.h>

/* Vectors are assumed to be in tangent space. */
namespace bxdf {

  ax_device_callable_inlined float absdot(const glm::vec3 &w1, const glm::vec3 &w2) { return glm::abs(glm::dot(w1, w2)); }

  ax_device_callable_inlined bool same_hemisphere(const glm::vec3 &w1, const glm::vec3 &w2) { return w1.z * w2.z > 0; }

  ax_device_callable_inlined float costheta(const glm::vec3 &v) { return v.z; }

  ax_device_callable_inlined float abscostheta(const glm::vec3 &v) { return fabsf(costheta(v)); }

  ax_device_callable_inlined float cos2theta(const glm::vec3 &v) { return math::sqr(costheta(v)); }

  ax_device_callable_inlined float cos2theta(float sintheta) { return 1.f - math::sqr(sintheta); }

  ax_device_callable_inlined float sin2theta(const glm::vec3 &v) { return std::max(0.f, 1.f - cos2theta(v)); }

  ax_device_callable_inlined float sin2theta(float costheta) { return std::max(0.f, 1.f - math::sqr(costheta)); }

  ax_device_callable_inlined float sintheta(const glm::vec3 &v) { return math::sqrt(sin2theta(v)); }

  ax_device_callable_inlined float sinphi(const glm::vec3 &v) {
    if (sintheta(v) == 0.f)
      return 0.f;
    return std::clamp(v.y / sintheta(v), -1.f, 1.f);
  }

  ax_device_callable_inlined float sin2phi(const glm::vec3 &v) { return math::sqr(sinphi(v)); }

  ax_device_callable_inlined float tantheta(const glm::vec3 &v) { return sintheta(v) / costheta(v); }

  ax_device_callable_inlined float tan2theta(const glm::vec3 &v) { return sin2theta(v) / cos2theta(v); }

  ax_device_callable_inlined float cosphi(const glm::vec3 &v) {
    if (sintheta(v) == 0.f)
      return 1.f;
    return std::clamp(v.x / sintheta(v), -1.f, 1.f);
  }
  /* wo points to the opposite side of the surface.*/
  ax_device_callable_inlined float refract(const glm::vec3 &wi, glm::vec3 n, float eta, bool &valid, glm::vec3 &wt) {
    AX_ASSERT_NEQ(eta, 0);
    valid = true;
    float costheta_i = glm::dot(wi, n);
    if (costheta_i < 0) {
      n = -n;
      eta = 1.f / eta;
      costheta_i = -costheta_i;
    }
    float sin2theta_i = sin2theta(costheta_i);
    float sin2theta_t = sin2theta_i / math::sqr(eta);
    if (sin2theta_t >= 1.f) {
      valid = false;
      return {};
    }
    float costheta_t = math::sqrt(1.f - sin2theta_t);

    wt = -wi / eta + (costheta_i / eta - costheta_t) * n;
    return eta;
  }

  ax_device_callable_inlined float cos2phi(const glm::vec3 &v) { return math::sqr(cosphi(v)); }

  /* Provides a uniform mapping from square to disk .*/
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

  ax_device_callable_inlined glm::vec2 sample_uniform_disk_polar(const float u[2]) {
    float r = math::sqrt(u[0]);
    float t = 2.f * M_PI * u[1];
    return {r * cos(t), r * sin(t)};
  }

}  // namespace bxdf

/* Takes etaX values as eta_incident / eta_transmission.*/
class Fresnel {
  using fcomplex = math::fcomplex;

  math::fcomplex etac{};
  float etar{};

 public:
  Fresnel(float eta, float k) : etac(eta, k) {}

  Fresnel(float eta) : etar(eta) {}

  /*
   * Dielectrics, and refractive materials.
   * new_etha is the computed eta value depending on costheta_i.
   * Assumes costheta_i = dot(wo , n)
   */
  ax_device_callable_inlined float real(float costheta_i, float &new_eta) {
    float eta = etar;
    costheta_i = glm::clamp(costheta_i, -1.f, 1.f);
    if (costheta_i < 0) {
      eta = 1.f / eta;
      costheta_i = -costheta_i;
    }
    new_eta = eta;
    float sin2theta_i = bxdf::sin2theta(costheta_i);
    float sin2theta_t = sin2theta_i / math::sqr(eta);
    if (sin2theta_t >= 1)
      return 1.f;
    float costheta_t = math::sqrt(1.f - sin2theta_t);
    float rpar = (eta * costheta_i - costheta_t) / (eta * costheta_i + costheta_t);
    float rperp = (costheta_i - eta * costheta_t) / (costheta_i + eta * costheta_t);
    return (math::sqr(rpar) + math::sqr(rperp)) / 2.f;
  }

  // Conductors and absorbant materials.
  ax_device_callable_inlined float complex(float costheta_i) {
    fcomplex eta = etac;
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
class NDF {
  float alpha_x, alpha_x_inv;
  float alpha_y, alpha_y_inv;
  static constexpr float PERFECT_SPECULAR_THRESHOLD = 1e-3f;
  ax_device_callable_inlined float roughnessToAlpha(float roughness) const {
    return roughness != 0 ? roughness * roughness : PERFECT_SPECULAR_THRESHOLD;
  }

 public:
  /**
   * @brief Creates an isotropic NDF with alpha_x = alpha_y = 1.f.
   *
   * @param roughness
   */
  ax_device_callable_inlined NDF(float roughness) {
    alpha_x = roughnessToAlpha(roughness);
    alpha_y = alpha_x;
    alpha_x_inv = 1.f / alpha_x;
    alpha_y_inv = 1.f / alpha_y;
  }

  /**
   * @brief Creates an anisotropic NDF with different alpha_x and alpha_y values.
   *
   * @param anisotropy Ratio (alpha_x/alpha_y) x anisotropy² coefficient.
   * @param roughness Roughness value.
   */
  ax_device_callable_inlined NDF(float anisotropy, float roughness) {
    anisotropy = std::clamp(anisotropy, -1.f, 1.f);
    float aspect = math::sqrt(1.f - 0.9f * anisotropy);

    AX_ASSERT_NEQ(aspect, 0.f);
    alpha_x = roughnessToAlpha(roughness) / aspect;
    alpha_y = roughnessToAlpha(roughness) * aspect;

    alpha_x_inv = 1.f / alpha_x;
    alpha_y_inv = 1.f / alpha_y;
  }
  /*
   * Anisotropic micrafacets distribution function.
   * Takes a half-vector or a shading normal.
   */
  ax_device_callable_inlined float D(const glm::vec3 &wm) const {
    AX_ASSERT_NEQ(alpha_x, 0.f);
    AX_ASSERT_NEQ(alpha_y, 0.f);
    if (bxdf::costheta(wm) < 1e-6f)
      return 0.f;

    float cos2phi_alpha2_x = bxdf::cos2phi(wm) * math::sqr(alpha_x_inv);
    float sin2phi_alpha2_y = bxdf::sin2phi(wm) * math::sqr(alpha_y_inv);
    float denom = M_PI * alpha_x * alpha_y * math::sqr(bxdf::cos2theta(wm)) *
                  math::sqr(1.f + bxdf::tan2theta(wm) * (cos2phi_alpha2_x + sin2phi_alpha2_y));
    AX_ASSERT_NEQ(denom, 0.f);
    return 1.f / denom;
  }

  ax_device_callable_inlined float lambda(const glm::vec3 &v) const {
    float tan2theta = bxdf::tan2theta(v);
    if (ISINF(tan2theta) || ISNAN(tan2theta))
      return 0;
    float alpha_x_term = math::sqr(alpha_x) * bxdf::cos2phi(v);
    float alpha_y_term = math::sqr(alpha_y) * bxdf::sin2phi(v);
    float alpha2 = alpha_x_term + alpha_y_term;
    return (math::sqrt(1 + alpha2 * tan2theta) - 1) * 0.5f;
  }

  /* Masking function : Statistically describes occlusions and backfacing microfacets.
   * Specifies the fraction of microfacets visible from direction v.
   */
  ax_device_callable_inlined float G1(glm::vec3 v) const {
    v.z = std::max(v.z, 1e-4f);
    return 1.f / (1.f + lambda(v));
  }

  /* Bi-Directional Masking-Shadowing function : Gives the amount of microfacets simultaneously visible from wo and wi.*/
  ax_device_callable_inlined float G(const glm::vec3 &wo, const glm::vec3 wi) const { return 1.f / (1.f + lambda(wo) + lambda(wi)); }

  ax_device_callable_inlined float D(const glm::vec3 &wo, const glm::vec3 &wm) const {
    return (G1(wo) / bxdf::abscostheta(wo)) * D(wm) * bxdf::absdot(wo, wm);
  }
  ax_device_callable_inlined float pdf(const glm::vec3 &wo, const glm::vec3 &wm) const { return D(wo, wm); }

  /* Returns a visible microfacet normal.*/
  ax_device_callable_inlined glm::vec3 sampleGGXVNDF(const glm::vec3 &wo, const float uc[2]) const {

    // Scales w to the alpha ellipsoid.
    glm::vec3 wo_h = glm::normalize(glm::vec3(wo.x * alpha_x, wo.y * alpha_y, wo.z));
    if (wo_h.z < 0)
      wo_h = -wo_h;

    // Create orthonormal frame.
    glm::vec3 t, b;
    if (wo_h.z < 1.f)
      t = glm::normalize(glm::cross(glm::vec3(0, 0, 1), wo_h));
    else
      t = glm::vec3(1, 0, 0);
    b = glm::cross(wo_h, t);

    // Sample two uniformly distributesd points on a disk.
    glm::vec2 polar_samples = bxdf::sample_uniform_disk_polar(uc);
    // Parameterization of projected area
    float s = 0.5f * (1.f + wo_h.z);
    polar_samples.y = math::lerp(math::sqrt(1.f - math::sqr(polar_samples.x)), polar_samples.y, s);
    glm::vec3 nh = polar_samples.x * t + polar_samples.y * b +
                   math::sqrt(fmax(0.f, 1.f - math::sqr(polar_samples.x) - math::sqr(polar_samples.y))) * wo_h;
    return glm::normalize(glm::vec3(alpha_x * nh.x, alpha_y * nh.y, std::max(1e-6f, nh.z)));
  }

  /* Checks if roughness allows for a specular lobe or a perfect specular effect.*/
  ax_device_callable_inlined bool isFullSpecular() const { return std::max(alpha_x, alpha_y) <= PERFECT_SPECULAR_THRESHOLD; }
};

#endif
