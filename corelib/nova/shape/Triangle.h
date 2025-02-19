#ifndef TRIANGLE_H
#define TRIANGLE_H
#include "MeshContext.h"
#include "internal/device/gpgpu/device_utils.h"
#include "ray/Hitable.h"
#include "shape_datastructures.h"
#include <internal/common/math/math_utils.h>
#include <internal/geometry/BoundingBox.h>
#include <internal/geometry/Object3D.h>
#include <internal/macro/project_macros.h>
namespace nova::shape {

  struct mesh_properties_t {
    const Object3D *geometry;
    std::size_t transform_offset;
  };

  class Triangle {
   private:
    /* Index of the triangle in the corresponding mesh_id's geometry.
     * Vertices are accessed with triangle_id + 0 , triangle_id + 1 ,triangle_id + 2 in the index array.
     */
    uint32_t triangle_id{};
    uint32_t mesh_id{};

   public:
    CLASS_DCM(Triangle)
    ax_device_callable Triangle(std::size_t mesh_id, std::size_t triangle_id);
    ax_device_callable transform::transform4x4_t getTransform(const MeshCtx &mesh_geometry) const;
    ax_device_callable bool hit(const Ray &ray, float tmin, float tmax, hit_data &data, const MeshCtx &mesh_geometry) const;
    ax_device_callable ax_no_discard geometry::BoundingBox computeAABB(const MeshCtx &mesh_geometry) const;
    ax_device_callable ax_no_discard glm::vec3 centroid(const MeshCtx &mesh_geometry) const;
    ax_device_callable ax_no_discard bool hasValidTangents(const MeshCtx &mesh_geometry) const;
    ax_device_callable ax_no_discard bool hasValidBitangents(const MeshCtx &mesh_geometry) const;
    ax_device_callable ax_no_discard bool hasValidNormals(const MeshCtx &mesh_geometry) const;
    ax_device_callable ax_no_discard bool hasValidUvs(const MeshCtx &mesh_geometry) const;

   private:
    ax_device_callable geometry::face_data_tri getFace(const MeshCtx &mesh_geometry) const;
    ax_device_callable geometry::face_data_tri getTransformedFace(const MeshCtx &mesh_geometry) const;
  };
}  // namespace nova::shape

#endif  // TRIANGLE_H
