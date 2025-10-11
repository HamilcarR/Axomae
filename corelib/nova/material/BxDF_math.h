#ifndef BXDF_MATH_H
#define BXDF_MATH_H

#include <internal/common/math/math_spherical.h>
#include <internal/device/gpgpu/device_macros.h>
#include <internal/device/gpgpu/device_utils.h>

/* Vectors are assumed to belong to a unit hemisphere. */

namespace bxdf {

  ax_device_callable_inlined float costheta(const glm::vec3 &angle) { return fabsf(angle.z); }

  ax_device_callable_inlined glm::vec3 hemisphere_sample_uniform(const float u[2]) {
    glm::vec2 sph = math::spherical::uvToSpherical(u[0], u[1]);
    return math::spherical::sphericalToCartesian(sph);
  }

  ax_device_callable_inlined constexpr float hemisphere_pdf() { return INV_2PI; }

}  // namespace bxdf
#endif
