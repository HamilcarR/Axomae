#include "shape_datastructures.h"
#include <glm/gtc/type_ptr.hpp>
#include <internal/macro/project_macros.h>
namespace nova::shape {
  namespace transform {

    ax_device_callable std::size_t get_transform_offset(std::size_t mesh_index, const mesh_transform_views_t &transform_views) {
      if (mesh_index >= transform_views.mesh_offsets_to_matrix.size())
        return INVALID_OFFSET;
      return transform_views.mesh_offsets_to_matrix[mesh_index];
    }

    ax_device_callable int reconstruct_transform4x4(transform4x4_t &ret_transform,
                                                    std::size_t flat_matrix_buffer_offset,
                                                    const mesh_transform_views_t &transform_views) {
      if (flat_matrix_buffer_offset >= transform_views.matrix_components_view.size()) {
        return -1;
      }
      constexpr int padding = transform4x4_t::padding();
      if (flat_matrix_buffer_offset % padding != 0) {
        return 1;
      }
      auto &storage = transform_views.matrix_components_view;
      memcpy(glm::value_ptr(ret_transform.m), &storage[flat_matrix_buffer_offset], 16 * sizeof(float));
      memcpy(glm::value_ptr(ret_transform.inv), &storage[flat_matrix_buffer_offset + 16], 16 * sizeof(float));
      memcpy(glm::value_ptr(ret_transform.t), &storage[flat_matrix_buffer_offset + 32], 16 * sizeof(float));
      memcpy(glm::value_ptr(ret_transform.n), &storage[flat_matrix_buffer_offset + 48], 9 * sizeof(float));

      return 0;
    }
  }  // namespace transform

  ax_device_callable MeshBundleViews::MeshBundleViews(const transform::mesh_transform_views_t &t_view,
                                                      const triangle::mesh_vertex_attrib_views_t &geom)
      : transforms(t_view), geometry(geom) {}

  void MeshBundleViews::set(const transform::mesh_transform_views_t &t, const triangle::mesh_vertex_attrib_views_t &g) {
    transforms = t;
    geometry = g;
  }

  ax_device_callable transform::transform4x4_t MeshBundleViews::reconstructTransform4x4(std::size_t mesh_index) const {
    AX_ASSERT_LT(mesh_index, transforms.mesh_offsets_to_matrix.size());
    std::size_t transform_offset = transform::get_transform_offset(mesh_index, transforms);
    AX_ASSERT_NEQ(transform_offset, transform::INVALID_OFFSET);
    transform::transform4x4_t returned_transform4x4{};
    int err = transform::reconstruct_transform4x4(returned_transform4x4, transform_offset, transforms);
    AX_ASSERT(err == 0, "Error reconstructing transformation matrix");
    return returned_transform4x4;
  }

  ax_device_callable std::size_t MeshBundleViews::getTransformOffset(std::size_t mesh_index) const {
    return transform::get_transform_offset(mesh_index, transforms);
  }

}  // namespace nova::shape
