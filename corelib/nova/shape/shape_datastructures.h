#ifndef SHAPE_DATASTRUCTURES_H
#define SHAPE_DATASTRUCTURES_H
#include <glm/glm.hpp>
#include <internal/common/axstd/span.h>
#include <internal/geometry/Object3D.h>

namespace nova::shape {

  namespace triangle {
    struct mesh_vertex_attrib_views_t {
      axstd::span<const Object3D> device_geometry_view;
      axstd::span<const Object3D> host_geometry_view;
    };
  }  // namespace triangle

  namespace transform {

    constexpr std::size_t INVALID_OFFSET = std::numeric_limits<std::size_t>::max();

    struct transform4x4_t {
      glm::mat4 m;
      glm::mat4 inv;  // inverse
      glm::mat4 t;    // transpose
      glm::mat3 n;    // normal ( mat3(transpose(invert(m)) )
      ax_device_callable bool operator==(const transform4x4_t &other) const {
        // no need to compare the others , waste of cycles. If the other matrices are not equal, we raise this in the assert.
        bool equal = m == other.m;
        AX_ASSERT(!equal || (inv == other.inv && t == other.t && n == other.n), "Invalid transform matrix");
        return equal;
      }
      ax_device_callable static constexpr std::size_t padding() { return 57; }  // how many elements in the record
    };

    struct transform3x3_t {
      glm::mat3 m;
      glm::mat3 inv;
      glm::mat3 t;
      bool operator==(const transform3x3_t &other) const { return m == other.m; }
      static constexpr std::size_t padding() { return 27; }
    };

    struct mesh_transform_views_t {
      axstd::span<const float> matrix_components_view{};
      axstd::span<const std::size_t> mesh_offsets_to_matrix{};
      /* Note in case we want to add more shapes indices : Provide an offset for each indices array :
       * triangle indices offset , sphere indices offset , etc , and access matrix_components_view using that offset.
       */
    };
    ax_device_callable std::size_t get_transform_offset(std::size_t mesh_index, const mesh_transform_views_t &transform_views);
    /**
     * Returns :
     * - -1 if the offset provided is invalid (>= matrix_components_view.size).\n
     * - 1 if the offset has a wrong padding that doesn't conform with transform4x4_t's elements padding (57).\n
     * - 0 if transform succeeded.
     */
    ax_device_callable int reconstruct_transform4x4(transform4x4_t &ret_transform,
                                                    std::size_t flat_matrix_buffer_offset,
                                                    const mesh_transform_views_t &transform_views);

  }  // namespace transform

  class MeshBundleViews {
    transform::mesh_transform_views_t transforms{};
    triangle::mesh_vertex_attrib_views_t geometry{};
    /* Add more attrib views for each kind of geometry. */

   public:
    CLASS_DCM(MeshBundleViews)

    ax_device_callable MeshBundleViews(const transform::mesh_transform_views_t &transform_views,
                                       const triangle::mesh_vertex_attrib_views_t &geometry);

    ax_device_callable void set(const transform::mesh_transform_views_t &transforms, const triangle::mesh_vertex_attrib_views_t &geometry);

    /**
     * See comment for reconstruct_transform4x4()
     */
    ax_device_callable transform::transform4x4_t reconstructTransform4x4(std::size_t mesh_index) const;
    ax_device_callable std::size_t getTransformOffset(std::size_t mesh_index) const;
    ax_device_callable const triangle::mesh_vertex_attrib_views_t &getTriangleGeometryViews() const { return geometry; }
    ax_device_callable const transform::mesh_transform_views_t &getTransforms() const { return transforms; }
    ax_device_callable Object3D getTriangleMesh(std::size_t mesh_id) const;
  };

}  // namespace nova::shape
#endif
