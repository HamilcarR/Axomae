#ifndef SHAPEINTERFACE_H
#define SHAPEINTERFACE_H
#include "Box.h"
#include "MeshContext.h"
#include "Sphere.h"
#include "Square.h"
#include "Triangle.h"
#include "internal/device/gpgpu/device_macros.h"
#include "ray/Hitable.h"
#include "shape_datastructures.h"
#include <internal/geometry/BoundingBox.h>
#include <internal/memory/tag_ptr.h>

namespace nova::shape {

  /* All methods return world space positions. */
  class NovaShapeInterface : public core::tag_ptr<Triangle, Sphere, Square, Box> {
   public:
    using tag_ptr::tag_ptr;

    ax_device_callable_inlined glm::vec3 centroid(const MeshCtx &geometry) const {
      auto d = [&](auto shape) { return shape->centroid(geometry); };
      return dispatch(d);
    }
    ax_device_callable_inlined geometry::BoundingBox computeAABB(const MeshCtx &geometry) const {
      auto d = [&](auto shape) { return shape->computeAABB(geometry); };
      return dispatch(d);
    }
    ax_device_callable_inlined float area(const MeshCtx &geometry) const {
      auto d = [&](auto shape) { return shape->area(geometry); };
      return dispatch(d);
    }

    /* Takes a world space ray , and fills data with world space positions , normals etc. */
    ax_device_callable_inlined bool hit(const Ray &r, float tmin, float tmax, hit_data &data, const MeshCtx &geometry) const {
      auto d = [&](auto shape) { return shape->hit(r, tmin, tmax, data, geometry); };
      return dispatch(d);
    }

    ax_device_callable_inlined face_data_s getFace(const MeshCtx &geometry) const {
      auto d = [&](auto shape) { return shape->getFace(geometry); };
      return dispatch(d);
    }

    ax_device_callable_inlined transform4x4_t getTransform(const MeshCtx &geometry) const {
      auto d = [&](auto shape) { return shape->getTransform(geometry); };
      return dispatch(d);
    }

    ax_device_callable_inlined const float *getTransformAddr(const MeshCtx &geometry) const {
      auto d = [&](auto shape) { return shape->getTransformAddr(geometry); };
      return dispatch(d);
    }
  };

  using TYPELIST = NovaShapeInterface::type_pack;
}  // namespace nova::shape
#endif  // SHAPEINTERFACE_H
