#ifndef SHAPEINTERFACE_H
#define SHAPEINTERFACE_H
#include "internal/geometry/BoundingBox.h"
#include "internal/memory/tag_ptr.h"
#include "ray/Hitable.h"

namespace nova::shape {
  class Sphere;
  class Square;
  class Triangle;
  class Box;

  class NovaShapeInterface : public core::tag_ptr<Triangle, Sphere, Square, Box> {
   public:
    using tag_ptr::tag_ptr;
    ax_device_callable ax_no_discard glm::vec3 centroid() const;
    ax_device_callable ax_no_discard geometry::BoundingBox computeAABB() const;
    ax_device_callable ax_no_discard bool hit(const Ray &r, float tmin, float tmax, hit_data &data, base_options *user_options) const;
  };

  using TYPELIST = NovaShapeInterface::type_pack;
}  // namespace nova::shape
#endif  // SHAPEINTERFACE_H
