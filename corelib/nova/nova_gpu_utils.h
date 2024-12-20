#ifndef NOVA_GPU_UTILS_H
#define NOVA_GPU_UTILS_H

#include "internal/common/axstd/span.h"
#include "internal/device/gpgpu/device_utils.h"
#include <cstdint>
#include <vector>

#ifdef AXOMAE_USE_CUDA
#  include "gpu/GPURandomGenerator.h"
#endif

namespace core::memory {
  template<class T>
  class MemoryArena;
}
/* defines functions and interfaces for nova to communicate with the device */
namespace nova {
  using cache_collection_t = std::vector<axstd::span<uint8_t>>;

  /* Tracks memory pool buffers that are potentially shared with devices.
   * Is cleared before scene loading , and rebuilt after.
   */
  struct device_shared_caches_t {
    /* Takes multiple buffers containing collections of objects... For ex :
     * index 0 : All textures raw data
     * index 1 : All geometry for mesh
     * ...
     */
    cache_collection_t contiguous_caches;

    void addSharedCacheAddress(axstd::span<uint8_t> buffer) { contiguous_caches.push_back(buffer); }
    void clear() { contiguous_caches.clear(); }
  };
}  // namespace nova

namespace nova::gputils {

  struct domain2d {
    unsigned x;
    unsigned y;
  };

#ifdef AXOMAE_USE_CUDA
  struct gpu_random_generator_t {
    math::random::GPUPseudoRandomGenerator xorshift;
    math::random::GPUQuasiRandomGenerator sobol;
  };
  struct gpu_util_structures_t {
    gpu_random_generator_t random_generator;
    kernel_argpack_t threads_distribution;
  };
#else
  struct gpu_random_generator_t {};
  struct gpu_util_structures_t {};

#endif

  gpu_util_structures_t initialize_gpu_structures(const domain2d &domain, core::memory::MemoryArena<std::byte> &arena);
  void cleanup_gpu_structures(gpu_util_structures_t &gpu_structures, core::memory::MemoryArena<std::byte> &arena);

  void clean_generators(gpu_random_generator_t &generators);

  void lock_host_memory_default(const device_shared_caches_t &collection);
  void unlock_host_memory(const device_shared_caches_t &collection);

}  // namespace nova::gputils
#endif  // NOVA_GPU_UTILS_H
