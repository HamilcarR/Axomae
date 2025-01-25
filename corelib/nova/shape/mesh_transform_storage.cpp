#include "mesh_transform_storage.h"
#include <internal/common/math/math_utils.h>
#include <internal/device/gpgpu/device_transfer_interface.h>
#include <limits>
namespace nova::shape::transform {
#ifdef AXOMAE_USE_CUDA
  constexpr bool is_gpu_build = true;
#else
  constexpr bool is_gpu_build = false;
#endif

  class Storage::GPUImpl {
    using GPUSharedBuffer = device::gpgpu::DeviceSharedBufferView;
    GPUSharedBuffer shared_buffer;

   public:
    template<class T>
    void addBuffer(axstd::span<T> &lockable_memory_area) {
      if constexpr (is_gpu_build)
        shared_buffer = GPUSharedBuffer(lockable_memory_area.data(), lockable_memory_area.size());
    }

    void map() {
      if constexpr (is_gpu_build) {
        auto query_result = pin_host_memory(shared_buffer, device::gpgpu::PIN_MODE_RO);
        DEVICE_ERROR_CHECK(query_result.error_status);
      }
    }

    void unmap() {
      if constexpr (is_gpu_build) {
        auto query_result = unpin_host_memory(shared_buffer);
        DEVICE_ERROR_CHECK(query_result.error_status);
      }
    }
  };
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

  Storage::Storage(bool use_gpu_) {
    using_gpu = use_gpu_ && is_gpu_build;
    device_memory_management = using_gpu ? std::make_unique<Storage::GPUImpl>() : nullptr;
  }

  Storage::~Storage() = default;
  Storage &Storage::operator=(Storage &&other) noexcept = default;
  Storage::Storage(Storage &&other) noexcept = default;

  void Storage::init(std::size_t total_meshe_size) { transform_lookup.offsets.resize(total_meshe_size); }

  static std::size_t push_packed_matrix_components(const glm::mat4 &mat, std::vector<float> &matrices) {
    std::size_t old_max_offset = matrices.size();
    for (int i = 0; i < 4; i++) {
      matrices.push_back(mat[i].x);
      matrices.push_back(mat[i].y);
      matrices.push_back(mat[i].z);
      matrices.push_back(mat[i].w);
    }
    return old_max_offset;
  }

  static std::size_t push_packed_matrix_components(const glm::mat3 &mat, std::vector<float> &matrices) {
    std::size_t old_max_offset = matrices.size();
    for (int i = 0; i < 3; i++) {
      matrices.push_back(mat[i].x);
      matrices.push_back(mat[i].y);
      matrices.push_back(mat[i].z);
    }
    return old_max_offset;
  }

  transform_hash_t Storage::getTransformOffset(const glm::mat4 &transform) const {
    auto &transform_table = transform_lookup.transfom_table;
    transform4x4_t temporary_transform;
    temporary_transform.m = transform;
    std::size_t h = hash(temporary_transform);
    auto it = transform_table.find(h);
    if (it != transform_table.end())
      return {h, it->second};
    return {h, INVALID_OFFSET};
  }

  std::size_t Storage::getTransformOffset(std::size_t mesh_index) const {
    if (mesh_index >= transform_lookup.offsets.size())
      return INVALID_OFFSET;
    return transform_lookup.offsets[mesh_index];
  }

  bool Storage::reconstructTransform4x4(transform4x4_t &transform, std::size_t elements_offset) const {
    if (elements_offset >= matrix_storage.elements.size()) {
      LOG("Invalid matrix element offset provided.", LogLevel::ERROR);
      return false;
    }
    constexpr int padding = transform4x4_t::padding();
    if (elements_offset % padding != 0) {
      LOG("Offset doesn't respect transform4x4_t padding requirement (" + std::to_string(padding * sizeof(float)) + " Bytes , has " +
              std::to_string(elements_offset * sizeof(float)) + " Bytes.",
          LogLevel::ERROR);
      return false;
    }
    auto &storage = matrix_storage.elements;
    std::memcpy(glm::value_ptr(transform.m), &storage[elements_offset], 16 * sizeof(float));
    std::memcpy(glm::value_ptr(transform.inv), &storage[elements_offset + 16], 16 * sizeof(float));
    std::memcpy(glm::value_ptr(transform.t), &storage[elements_offset + 32], 16 * sizeof(float));
    std::memcpy(glm::value_ptr(transform.n), &storage[elements_offset + 48], 9 * sizeof(float));
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
      matrices.n = glm::mat3(glm::transpose(matrices.inv));

      std::size_t matrix_element_offset = push_packed_matrix_components(matrices.m, matrix_storage.elements);
      push_packed_matrix_components(matrices.inv, matrix_storage.elements);
      push_packed_matrix_components(matrices.t, matrix_storage.elements);
      push_packed_matrix_components(matrices.n, matrix_storage.elements);
      transform_lookup.offsets[mesh_index] = matrix_element_offset;
      transform_lookup.transfom_table[transform_hash.hash] = matrix_element_offset;
    }
  }

  void Storage::updateViews() {
    if (is_mapped)
      LOG("Transform views buffers changed while storage is still mapped.", LogLevel::WARNING);
    matrix_storage.elements_view = axstd::span(matrix_storage.elements.data(), matrix_storage.elements.size());
    transform_lookup.offsets_view = axstd::span(transform_lookup.offsets.data(), transform_lookup.offsets.size());
  }

  void Storage::map() {
    updateViews();
    device_memory_management->addBuffer(matrix_storage.elements_view);
    device_memory_management->addBuffer(transform_lookup.offsets_view);
    device_memory_management->map();
    is_mapped = true;
  }

  void Storage::unmap() {
    device_memory_management->unmap();
    is_mapped = false;
  }

  void Storage::clear() {
    matrix_storage.elements.clear();
    transform_lookup.offsets.clear();
    matrix_storage.elements_view = axstd::span<float>();
    transform_lookup.offsets_view = axstd::span<std::size_t>();
  }

}  // namespace nova::shape::transform