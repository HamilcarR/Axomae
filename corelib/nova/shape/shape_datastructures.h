#ifndef SHAPE_DATASTRUCTURES_H
#define SHAPE_DATASTRUCTURES_H
#include "glm/gtc/type_ptr.hpp"
#include <internal/common/axstd/span.h>
#include <internal/common/math/math_includes.h>
#include <internal/common/math/utils_3D.h>
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

    struct mesh_transform_views_t {
      axstd::span<const float> matrix_components_view{};
      axstd::span<const uint64_t> mesh_offsets_to_matrix{};
      /* Note in case we want to add more shapes indices : Provide an offset for each indices array :
       * triangle indices offset , sphere indices offset , etc , and access matrix_components_view using that offset.
       */
    };

    ax_device_callable_inlined uint64_t get_transform_offset(uint64_t mesh_index, const mesh_transform_views_t &transform_views) {
      if (mesh_index >= transform_views.mesh_offsets_to_matrix.size())
        return INVALID_OFFSET;
      return transform_views.mesh_offsets_to_matrix[mesh_index];
    }

    /**
     * Returns :
     * - -1 if the offset provided is invalid (>= matrix_components_view.size).\n
     * - 1 if the offset has a wrong padding that doesn't conform with transform4x4_t's elements padding (57).\n
     * - 0 if transform succeeded.
     */
    ax_device_callable_inlined int reconstruct_transform4x4(transform4x4_t &ret_transform,
                                                            uint64_t flat_matrix_buffer_offset,
                                                            const mesh_transform_views_t &transform_views) {
      if (flat_matrix_buffer_offset >= transform_views.matrix_components_view.size()) {
        return -1;
      }
      constexpr int padding = transform4x4_t::padding();
      if (flat_matrix_buffer_offset % padding != 0) {
        return 1;
      }
      auto &storage = transform_views.matrix_components_view;
      for (size_t i = 0; i < 16; i++) {
        glm::value_ptr(ret_transform.m)[i] = storage[flat_matrix_buffer_offset + i];
        glm::value_ptr(ret_transform.inv)[i] = storage[flat_matrix_buffer_offset + 16 + i];
        glm::value_ptr(ret_transform.t)[i] = storage[flat_matrix_buffer_offset + 32 + i];
      }
      for (size_t i = 0; i < 9; i++) {
        glm::value_ptr(ret_transform.n)[i] = storage[flat_matrix_buffer_offset + 48 + i];
      }

      return 0;
    }

  }  // namespace transform

  enum FACE { TRIANGLE, BOX, SQUARE, SPHERE };

  struct face_data_s {
    FACE type;
    union {
      geometry::face_data_tri triangle_face;
      float sphere_r;
      // TODO: Add other faces types here later.
    } data;
  };

}  // namespace nova::shape
#endif
