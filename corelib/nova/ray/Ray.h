#ifndef RAY_H
#define RAY_H
#include <internal/common/math/math_utils.h>
#include <internal/device/gpgpu/device_macros.h>
#include <internal/macro/project_macros.h>

namespace nova {
  class Ray {

    static constexpr float eps = 1e-3f;

   public:
    glm::vec3 origin{};
    glm::vec3 direction{};
    float tnear{0.000001f}, tfar{1e30f};

    ax_device_callable_inlined Ray() = default;
    ax_device_callable_inlined Ray(const glm::vec3 &o, const glm::vec3 &d) : origin(o), direction(d) {}
    ax_device_callable_inlined explicit Ray(const glm::vec3 &d) : origin(0), direction(d) {}
    ax_device_callable_inlined glm::vec3 pointAt(float t) const { return origin + t * direction; }

    ax_device_callable_inlined static Ray spawn(const glm::vec3 &wi, const glm::vec3 &n, const glm::vec3 &position, float epsilon = eps) {
      Ray ray;
      ray.direction = glm::normalize(wi);
      ray.origin = position + n * (epsilon + epsilon * glm::sign(glm::dot(n, wi)));
      return ray;
    }
  };

  /* For AA */
  class RayDifferential : public Ray {};
}  // namespace nova
#endif
