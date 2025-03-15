#ifndef BOX_H
#define BOX_H
#include "MeshContext.h"
#include "internal/device/gpgpu/device_utils.h"
#include "internal/macro/project_macros.h"
#include "ray/Hitable.h"
#include "ray/Ray.h"
#include <internal/geometry/BoundingBox.h>

namespace nova::shape {
  class Box {
   private:
    geometry::BoundingBox aabb;

   public:
    CLASS_DCM(Box)

    ax_device_callable Box(const glm::vec3 &min_coords, const glm::vec3 &max_coords);
    ax_device_callable explicit Box(const float *vertices, std::size_t size);
    ax_device_callable explicit Box(const geometry::BoundingBox &aabb);
    ax_host_only explicit Box(const std::vector<float> &vertices);

    /**
     * @brief tmin must be initialized to a small value (0.f) while tmax should be set at a highest value.
     * Use camera near and far .
     */
    ax_device_callable ax_no_discard bool hit(const Ray &ray, float tmin, float tmax, hit_data &data, const MeshCtx &mesh_geometry) const;
    ax_device_callable ax_no_discard glm::vec3 centroid(const MeshCtx & /*mesh_geometry*/) const { return aabb.getPosition(); }
    ax_device_callable ax_no_discard const glm::vec3 &getPosition() const { return aabb.getPosition(); }
    ax_device_callable ax_no_discard geometry::BoundingBox computeAABB(const MeshCtx & /*mesh_geometry*/) const { return aabb; }
    ax_device_callable ax_no_discard geometry::BoundingBox computeAABB() const { return aabb; }
    ax_device_callable ax_no_discard float area(const MeshCtx &) const { return 2.f * aabb.halfArea(); }
  };
}  // namespace nova::shape

#endif  // BOX_H
