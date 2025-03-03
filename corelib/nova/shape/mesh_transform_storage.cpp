#include "mesh_transform_storage.h"
#include "shape_datastructures.h"
#include <internal/common/math/math_utils.h>
#include <internal/common/math/utils_3D.h>
#include <internal/debug/Logger.h>
namespace nova::shape::transform {

  /*******************************************************************************************************************************************/
  static void hash_matrix4x4(std::size_t &seed, const glm::mat4 &matrix) {
    const float *f = glm::value_ptr(matrix);
    for (int i = 0; i < 16; i++)
      math::hash(seed, f[i]);
  }

  std::size_t hash(const transform4x4_t &transform) {
    std::size_t seed = 0x485D442;
    const glm::mat4 m = transform.m;
    hash_matrix4x4(seed, m);
    return seed;
  }

  Storage::Storage(bool is_using_gpu) { store_vram = is_using_gpu && core::build::is_gpu_build; }

  void Storage::init(std::size_t total_meshe_size) { transform_lookup.offsets.resize(total_meshe_size); }

  static std::size_t push_packed_matrix_components(const glm::mat4 &mat, axstd::managed_vector<float> &matrices) {
    std::size_t old_max_offset = matrices.size();
    for (int i = 0; i < 4; i++) {
      matrices.push_back(mat[i].x);
      matrices.push_back(mat[i].y);
      matrices.push_back(mat[i].z);
      matrices.push_back(mat[i].w);
    }
    return old_max_offset;
  }

  static std::size_t push_packed_matrix_components(const glm::mat3 &mat, axstd::managed_vector<float> &matrices) {
    std::size_t old_max_offset = matrices.size();
    for (int i = 0; i < 3; i++) {
      matrices.push_back(mat[i].x);
      matrices.push_back(mat[i].y);
      matrices.push_back(mat[i].z);
    }
    return old_max_offset;
  }

  transform_hash_t Storage::getTransformOffset(const glm::mat4 &transform) const {
    auto &transform_table = transform_lookup.transform_table;
    transform4x4_t temporary_transform;
    temporary_transform.m = transform;
    std::size_t h = hash(temporary_transform);
    auto it = transform_table.find(h);
    if (it != transform_table.end())
      return {h, it->second};
    return {h, INVALID_OFFSET};
  }

  std::size_t Storage::getTransformOffset(std::size_t mesh_index) const {
    mesh_transform_views_t mtv = getTransformViews();
    AX_ASSERT_FALSE(mtv.mesh_offsets_to_matrix.empty());
    return get_transform_offset(mesh_index, mtv);
  }

  bool Storage::reconstructTransform4x4(transform4x4_t &transform, std::size_t elements_offset) const {
    mesh_transform_views_t mtv = getTransformViews();
    int error = reconstruct_transform4x4(transform, elements_offset, mtv);
    if (error == -1) {
      LOG("Invalid matrix element offset provided.", LogLevel::ERROR);
      return false;
    }
    if (error == 1) {
      LOG("Offset doesn't respect transform4x4_t padding requirement (" + std::to_string(transform4x4_t::padding() * sizeof(float)) +
              " Bytes , has " + std::to_string(elements_offset * sizeof(float)) + " Bytes.",
          LogLevel::ERROR);
      return false;
    }
    return true;
  }

  void Storage::add(const glm::mat4 &transform, uint32_t mesh_index) {
    AX_ASSERT_TRUE(!transform_lookup.offsets.empty());
    if (transform_lookup.offsets.empty()) {
      LOG("Storage not initialized.", LogLevel::ERROR);
      return;
    }
    transform_hash_t transform_hash = getTransformOffset(transform);
    if (transform_hash.transform_offset != INVALID_OFFSET) {
      transform_lookup.offsets[mesh_index] = transform_hash.transform_offset;
    } else {
      transform4x4_t matrices{};
      matrices.m = transform;
      matrices.inv = glm::inverse(matrices.m);
      matrices.t = glm::transpose(matrices.m);
      matrices.n = math::geometry::compute_normal_mat(matrices.m);

      std::size_t matrix_element_offset = push_packed_matrix_components(matrices.m, matrix_storage.elements);
      push_packed_matrix_components(matrices.inv, matrix_storage.elements);
      push_packed_matrix_components(matrices.t, matrix_storage.elements);
      push_packed_matrix_components(matrices.n, matrix_storage.elements);
      transform_lookup.offsets[mesh_index] = matrix_element_offset;
      transform_lookup.transform_table[transform_hash.hash] = matrix_element_offset;
    }
  }

  mesh_transform_views_t Storage::getTransformViews() const {
    mesh_transform_views_t mtv;
    mtv.matrix_components_view = matrix_storage.elements;
    mtv.mesh_offsets_to_matrix = transform_lookup.offsets;
    return mtv;
  }

  void Storage::map() {
    mesh_transform_views_t mtv = getTransformViews();
    is_mapped = true;
  }

  void Storage::unmap() { is_mapped = false; }

  void Storage::clear() {
    matrix_storage.elements.clear();
    transform_lookup.offsets.clear();
    transform_lookup.transform_table.clear();
  }

}  // namespace nova::shape::transform
