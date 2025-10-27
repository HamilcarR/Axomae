#ifndef INTERSECTFRAME_H
#define INTERSECTFRAME_H
#include "glm/geometric.hpp"
#include <internal/common/math/utils_3D.h>
#include <internal/device/gpgpu/device_macros.h>
#include <internal/device/gpgpu/device_utils.h>

/**
 * @brief Provides conversions procedures between local shading space to world space.
 */
class IntersectFrame {
  glm::mat3 tbn{1.f};  // orthonormal frame.
  glm::mat3 tbn_t{1.f};

 public:
  ax_device_callable_inlined IntersectFrame() = default;

  /* Recomputes the frame to ensure perfect orthonormality (Gram-Shmidt).*/
  static ax_device_callable_inlined void orthonormalize(glm::vec3 &t, glm::vec3 &b, glm::vec3 &n) {
    n = glm::normalize(n);
    t = glm::normalize(t - glm::dot(t, n) * n);
    b = glm::normalize(glm::cross(n, t));
  }

  ax_device_callable_inlined IntersectFrame(const float tangent[3], const float bitangent[3], const float normal[3], bool normalize = false) {
    glm::vec3 n(normal[0], normal[1], normal[2]);
    glm::vec3 t(tangent[0], tangent[1], tangent[2]);
    glm::vec3 b(bitangent[0], bitangent[1], bitangent[2]);
    if (normalize)
      orthonormalize(t, b, n);
    tbn = glm::mat3(t, b, n);
    tbn_t = glm::transpose(tbn);
  }

  ax_device_callable_inlined IntersectFrame(const glm::vec3 &tangent,
                                            const glm::vec3 &bitangent,
                                            const glm::vec3 &normal,
                                            bool orthonormalize = false)
      : IntersectFrame((const float *)&tangent, (const float *)&bitangent, (const float *)&normal, orthonormalize) {}

  ax_device_callable_inlined IntersectFrame(glm::vec3 normal, glm::vec3 tangent) {
    glm::vec3 bitangent = {};
    orthonormalize(tangent, bitangent, normal);
    tbn = glm::mat3(tangent, bitangent, normal);
    tbn_t = glm::transpose(tbn);
  }

  ax_device_callable_inlined glm::vec3 localToWorld(const glm::vec3 &vect) const { return tbn * vect; }

  ax_device_callable_inlined glm::vec3 worldToLocal(const glm::vec3 &vect) const { return tbn_t * vect; }

  ax_device_callable_inlined const glm::mat3 &getFrame() const { return tbn; }

  ax_device_callable_inlined const glm::mat3 &getTranspose() const { return tbn_t; }

  ax_device_callable_inlined const glm::vec3 &getNormal() const { return tbn[2]; }

  ax_device_callable_inlined const glm::vec3 &getTangent() const { return tbn[0]; }

  ax_device_callable_inlined const glm::vec3 &getBitangent() const { return tbn[1]; }
};

#endif