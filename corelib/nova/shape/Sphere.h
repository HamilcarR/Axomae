#ifndef SPHERE_H
#define SPHERE_H
#include "MeshContext.h"
#include "internal/macro/project_macros.h"
#include "ray/Hitable.h"
#include "shape/shape_datastructures.h"
#include <internal/common/math/math_utils.h>
#include <internal/geometry/BoundingBox.h>

namespace nova::shape {
  class Sphere {
   private:
    float radius{};
    glm::vec3 origin{};

   public:
    CLASS_DCM(Sphere)

    ax_device_callable Sphere(const glm::vec3 &origin, float radius);
    ax_device_callable bool hit(const Ray &ray, float tmin, float tmax, hit_data &data, const MeshCtx &geometry) const;
    ax_device_callable ax_no_discard glm::vec3 centroid(const MeshCtx &geometry) const { return origin; }
    ax_device_callable ax_no_discard geometry::BoundingBox computeAABB(const MeshCtx &geometry) const;
    ax_device_callable ax_no_discard float area(const MeshCtx & /*geometry*/) const { return 2 * PI * radius; }
    ax_device_callable_inlined transform4x4_t getTransform(const MeshCtx &geometry) const {
      AX_UNREACHABLE;
      return {};
    }
    ax_device_callable_inlined const float *getTransformAddr(const MeshCtx &geometry) const {
      AX_UNREACHABLE;
      return {};
    }
    ax_device_callable face_data_s getFace(const MeshCtx & /*geometry*/) const {
      face_data_s face{};
      face.type = SPHERE;
      face.data.sphere_r = radius;
      return face;
    }
  };
}  // namespace nova::shape
#endif  // SPHERE_H
