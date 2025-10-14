#ifndef PLANE_H
#define PLANE_H
#include "internal/macro/project_macros.h"
#include "ray/Hitable.h"
#include "shape/MeshContext.h"
#include "shape/shape_datastructures.h"
#include <internal/geometry/BoundingBox.h>

namespace nova::shape {
  class Square {
   private:
    glm::vec3 origin{};
    glm::vec3 side_w{};
    glm::vec3 side_h{};
    glm::vec3 normal{};
    glm::vec3 center{};

   public:
    CLASS_DCM(Square)

    ax_device_callable explicit Square(const glm::vec3 &origin_);
    /**
     * side_w and side_h must be perpendicular.
     * Constructor will normalize the normal.
     **/
    ax_device_callable Square(const glm::vec3 &origin_, const glm::vec3 &side_w_, const glm::vec3 &side_h_);

    ax_device_callable bool hit(const Ray &ray, float tmin, float tmax, hit_data &data, const MeshCtx &geometry) const;
    ax_device_callable ax_no_discard glm::vec3 centroid(const MeshCtx &geometry) const { return center; }
    ax_device_callable ax_no_discard geometry::BoundingBox computeAABB(const MeshCtx &geometry) const;
    ax_device_callable ax_no_discard float area(const MeshCtx &geometry) const { return glm::length(side_w) * glm::length(side_h); }
    ax_device_callable_inlined transform4x4_t getTransform(const MeshCtx &geometry) const {
      AX_UNREACHABLE;
      return {};
    }
    ax_device_callable face_data_s getFace(const MeshCtx & /*geometry*/) const {
      AX_UNREACHABLE;
      return {};
    }
  };
}  // namespace nova::shape

#endif  // PLANE_H
