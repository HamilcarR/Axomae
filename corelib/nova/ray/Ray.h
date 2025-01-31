#ifndef RAY_H
#define RAY_H
#include <internal/common/math/math_utils.h>

#include <internal/macro/project_macros.h>
namespace nova {
  class Ray {
   public:
    glm::vec3 origin;
    glm::vec3 direction;

   public:
    CLASS_DCM(Ray)
    ax_device_callable Ray(const glm::vec3 &o, const glm::vec3 &d) : origin(o), direction(d) {}
    ax_device_callable explicit Ray(const glm::vec3 &d) : origin(0), direction(d) {}
    ax_device_callable ax_no_discard glm::vec3 pointAt(float t) const { return origin + t * direction; }
  };

  /* For AA */
  class RayDifferential : public Ray {};
}  // namespace nova
#endif
