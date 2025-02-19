#ifndef SHAPEINTERFACE_H
#define SHAPEINTERFACE_H
#include "Box.h"
#include "MeshContext.h"
#include "Sphere.h"
#include "Square.h"
#include "Triangle.h"
#include "ray/Hitable.h"
#include <internal/geometry/BoundingBox.h>
#include <internal/memory/tag_ptr.h>

namespace nova::shape {
  class NovaShapeInterface : public core::tag_ptr<Triangle, Sphere, Square, Box> {
   public:
    using tag_ptr::tag_ptr;
    ax_device_callable ax_no_discard glm::vec3 centroid(const MeshCtx &geometry) const;
    ax_device_callable ax_no_discard geometry::BoundingBox computeAABB(const MeshCtx &geometry) const;
    ax_device_callable ax_no_discard bool hit(const Ray &r, float tmin, float tmax, hit_data &data, const MeshCtx &geometry) const;
  };

  using TYPELIST = NovaShapeInterface::type_pack;
}  // namespace nova::shape
#endif  // SHAPEINTERFACE_H
