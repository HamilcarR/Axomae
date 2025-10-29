#ifndef BXDF_MATH_H
#define BXDF_MATH_H

#include <internal/common/math/math_spherical.h>
#include <internal/device/gpgpu/device_macros.h>
#include <internal/device/gpgpu/device_utils.h>
#include <math.h>

/* Vectors are assumed to belong to a unit hemisphere. */

namespace bxdf {

  ax_device_callable_inlined float costheta(const glm::vec3 &angle) { return fabsf(angle.z); }

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
#endif
