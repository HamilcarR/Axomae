#ifndef SPHERE_H
#define SPHERE_H
#include "internal/common/math/math_utils.h"
#include "internal/geometry/BoundingBox.h"
#include "ray/Hitable.h"

namespace nova::shape {
  class Sphere {
   private:
    float radius{};
    glm::vec3 origin{};

   public:
    CLASS_DCM(Sphere)

    ax_device_callable Sphere(const glm::vec3 &origin, float radius);
    ax_device_callable bool hit(const Ray &ray, float tmin, float tmax, hit_data &data, base_options *user_options) const;
    ax_device_callable ax_no_discard glm::vec3 centroid() const { return origin; }
    ax_device_callable ax_no_discard geometry::BoundingBox computeAABB() const;
  };
}  // namespace nova::shape
#endif  // SPHERE_H
