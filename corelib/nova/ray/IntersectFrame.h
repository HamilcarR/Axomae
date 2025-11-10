#ifndef INTERSECTFRAME_H
#define INTERSECTFRAME_H
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

  ax_device_callable_inlined IntersectFrame(const float tangent[3], const float bitangent[3], const float normal[3], bool normalize = false) {
    glm::vec3 n(normal[0], normal[1], normal[2]);
    glm::vec3 t(tangent[0], tangent[1], tangent[2]);

    glm::vec3 norm_n, norm_t, norm_b;
    if (normalize)
      math::bran_shmidt(n, t, norm_n, norm_t, norm_b);
    tbn = glm::mat3(norm_t, norm_b, norm_n);
    tbn_t = glm::transpose(tbn);
  }

  ax_device_callable_inlined IntersectFrame(const glm::vec3 &tangent,
                                            const glm::vec3 &bitangent,
                                            const glm::vec3 &normal,
                                            bool orthonormalize = false)
      : IntersectFrame((const float *)&tangent, (const float *)&bitangent, (const float *)&normal, orthonormalize) {}

  ax_device_callable_inlined IntersectFrame(glm::vec3 normal, glm::vec3 tangent) {
    glm::vec3 norm_n, norm_t, norm_b;
    math::bran_shmidt(normal, tangent, norm_n, norm_t, norm_b);
    tbn = glm::mat3(norm_t, norm_b, norm_n);
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