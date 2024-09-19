#ifndef SHAPE_DATASTRUCTURES_H
#define SHAPE_DATASTRUCTURES_H
#include "glm/gtc/type_ptr.hpp"
#include <internal/common/axstd/span.h>
#include <internal/common/math/math_includes.h>
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
