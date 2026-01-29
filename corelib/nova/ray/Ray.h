#ifndef RAY_H
#define RAY_H
#include <internal/common/math/math_utils.h>
#include <internal/device/gpgpu/device_macros.h>
#include <internal/macro/project_macros.h>

namespace nova {
  class Ray {

    static constexpr float eps = 1e-3f;
    /* See "A Fast and Robust Method for Avoiding Self-Intersection" , C. Wachter and N. Binder. */
    static ax_device_callable_inlined float int_offset(float f, float n_comp) {
      constexpr float origin = 1.0f / 32.0f;
      constexpr float float_scale = 1.0f / 65536.0f;
      constexpr float int_scale = 256.0f;

      int32_t i = *(int32_t *)&f;
      i += (f > 0) ? (n_comp > 0 ? 1 : -1) : (n_comp > 0 ? -1 : 1);
      float f_offset = *(float *)&i;
      return std::abs(f) < origin ? f + float_scale * n_comp : f_offset;
    }

   public:
    glm::vec3 origin{};
    glm::vec3 direction{};
    float tnear{0.000001f}, tfar{1e30f};

    ax_device_callable_inlined Ray() = default;
    ax_device_callable_inlined Ray(const glm::vec3 &o, const glm::vec3 &d) : origin(o), direction(d) {}
    ax_device_callable_inlined explicit Ray(const glm::vec3 &d) : origin(0), direction(d) {}
    ax_device_callable_inlined glm::vec3 pointAt(float t) const { return origin + t * direction; }

    ax_device_callable_inlined static Ray spawn(const glm::vec3 &wi, const glm::vec3 &n_geom, const glm::vec3 &p) {
      Ray ray;
      ray.direction = glm::normalize(wi);
      // Apply adaptive offset to each component independently
      glm::vec3 offset_p = glm::vec3(int_offset(p.x, n_geom.x), int_offset(p.y, n_geom.y), int_offset(p.z, n_geom.z));
      ray.origin = (glm::dot(n_geom, wi) > 0) ? offset_p : p - (offset_p - p);
      return ray;
    }
  };

  /* For AA */
  class RayDifferential : public Ray {};
}  // namespace nova
#endif
