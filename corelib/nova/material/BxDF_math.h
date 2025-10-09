#ifndef BXDF_MATH_H
#define BXDF_MATH_H

#include <internal/common/math/math_spherical.h>
#include <internal/device/gpgpu/device_utils.h>

/* Vectors are assumed to belong to a unit hemisphere. */

namespace bxdf {
  ax_device_callable_inlined float costheta(const glm::vec3 &angle) { return angle.z; }
  ax_device_callable_inlined glm::vec3 hemisphere_sample_uniform(float u[2]) {
    glm::vec2 sph = math::spherical::uvToSpherical(u[0], u[1]);
    return math::spherical::sphericalToCartesian(sph);
  }
}  // namespace bxdf
#endif
