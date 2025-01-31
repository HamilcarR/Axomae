#ifndef TRIANGLE_H
#define TRIANGLE_H
#include "internal/device/gpgpu/device_utils.h"
#include "ray/Hitable.h"
#include "shape_datastructures.h"
#include <internal/common/math/math_utils.h>
#include <internal/geometry/BoundingBox.h>
#include <internal/geometry/Object3D.h>
#include <internal/macro/project_macros.h>
namespace nova::shape {

  struct mesh_shared_views_t {
    transform::mesh_transform_views_t transforms;
    triangle::mesh_vertex_attrib_views_t geometry;
  };
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

    ax_device_callable mesh_properties_t getMesh() const;
    ax_device_callable transform::transform4x4_t getTransform() const;
    ax_host_only static void updateSharedBuffers(const mesh_shared_views_t &geo);
    ax_device_callable Triangle(std::size_t mesh_id, std::size_t triangle_id);
    ax_device_callable bool hit(const Ray &ray, float tmin, float tmax, hit_data &data, base_options *user_options) const;
    ax_device_callable ax_no_discard geometry::BoundingBox computeAABB() const;

    ax_device_callable ax_no_discard glm::vec3 centroid() const;
    ax_device_callable ax_no_discard bool hasValidTangents() const;
    ax_device_callable ax_no_discard bool hasValidBitangents() const;
    ax_device_callable ax_no_discard bool hasValidNormals() const;
    ax_device_callable ax_no_discard bool hasValidUvs() const;

   private:
    ax_device_callable geometry::face_data_tri getFace() const;
  };
}  // namespace nova::shape

#endif  // TRIANGLE_H
