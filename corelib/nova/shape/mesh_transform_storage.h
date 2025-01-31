#ifndef MESH_TRANSFORM_STORAGE_H
#define MESH_TRANSFORM_STORAGE_H

#include "shape_datastructures.h"
#include <boost/functional/hash.hpp>
#include <internal/common/axstd/managed_buffer.h>
#include <internal/common/axstd/span.h>
#include <internal/macro/project_macros.h>

/* Stores a unique transformation, for lower memory footprint . */
namespace nova::shape::transform {

  struct transform_hash_t {
    std::size_t hash;
    std::size_t transform_offset;
  };

  std::size_t hash(const transform4x4_t &obj);

  struct matrix_elem_storage_t {

    /**
     * Stores a unique transformation matrix , and its pre-computed inverse , transposed , and normal matrix.
     * The representation of data is column major.\n
     * elements[0] , elements[1] , elements[2] , elements[3] are the first column of m.\n
     * elements[16] , ... , elements[19] are the first column of inv.\n
     */
    // std::vector<float> elements;
    axstd::managed_vector<float> elements;
  };

  struct transform_lookup_t {
    /**
     * Keeps track of the transformation of each mesh using an offset on <code>matrix_elem_storage_t::elements</code> , ie \n
     * offsets[mesh_id] = mesh_id's transformation matrix offset.
     */
    axstd::managed_vector<std::size_t> offsets;

    /**
     * Stores every unique transformation hash as key , and the offset of the first element of m matrix in their transform4x4_t as value.
     */
    std::unordered_map<std::size_t, std::size_t> transfom_table;
  };

  class Storage {

    matrix_elem_storage_t matrix_storage;
    transform_lookup_t transform_lookup;
    std::size_t total_meshes{};
    bool is_mapped{false};
    bool store_vram{false};

   public:
    explicit Storage(bool attempt_gpu_storage = true);
    void init(std::size_t total_meshes);
    void add(const glm::mat4 &transform, uint32_t mesh_index);
    /* Will call updateViews() before mapping to update the different view buffers. */
    void map();
    void unmap();
    void clear();
    /* Returns the offset of the first element of the first matrix if the transformation exists or INVALID_OFFSET. */
    transform_hash_t getTransformOffset(const glm::mat4 &transform) const;
    std::size_t getTransformOffset(std::size_t mesh_index) const;

    /**
     * Reconstructs a transform4x4_t from the matrix_storage using an offset into the storage.
     * Returns false if reconstruction has failed.
     */
    bool reconstructTransform4x4(transform4x4_t &ret_transform, std::size_t flat_matrix_buffer_offset) const;

    mesh_transform_views_t getTransformViews() const;
  };
}  // namespace nova::shape::transform

#endif  // MESH_TRANSFORM_STORAGE_H
