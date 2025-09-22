#include "nova_gpu_utils.h"
#include <internal/device/gpgpu/device_transfer_interface.h>
#include <internal/device/gpgpu/device_utils.h>
#include <internal/memory/MemoryArena.h>

namespace resrc = device::gpgpu;
namespace nova::gputils {
#if defined(AXOMAE_USE_CUDA)
  void lock_host_memory_default(device_shared_caches_t &collection) {
    for (auto &element : collection.contiguous_caches)
      if (!element.locked) {
        DEVICE_ERROR_CHECK(resrc::pin_host_memory(element.buffer.data(), element.buffer.size(), device::gpgpu::PIN_MODE_DEFAULT).error_status);
        element.locked = true;
      }
  }

  void unlock_host_memory(device_shared_caches_t &collection) {
    for (auto &element : collection.contiguous_caches)
      if (element.locked) {
        DEVICE_ERROR_CHECK(resrc::unpin_host_memory(element.buffer.data()).error_status);
        element.locked = false;
      }
  }

  static gpu_random_generator_t initialize_rand() {
    gpu_random_generator_t gpu_generators;
    gpu_generators.sobol.allocDeviceLookupTable();
    return gpu_generators;
  }

  void clean_generators(gpu_random_generator_t &generators) { generators.sobol.deallocDeviceLookupTable(); }

  gpu_util_structures_t initialize_gpu_structures() {
    gpu_util_structures_t gpu_structs;
    gpu_structs.random_generator = initialize_rand();
    return gpu_structs;
  }

  void cleanup_gpu_structures(gpu_util_structures_t &gpu_structures) { clean_generators(gpu_structures.random_generator); }

#else
  void lock_host_memory_default(device_shared_caches_t &) { EMPTY_FUNCBODY }
  void unlock_host_memory(device_shared_caches_t &) { EMPTY_FUNCBODY }

  void clean_generators(gpu_random_generator_t &) { EMPTY_FUNCBODY }

  void cleanup_gpu_structures(gpu_util_structures_t &) { EMPTY_FUNCBODY }

  void initialize_gpu_structures() { EMPTY_FUNCBODY }

#endif
}  // namespace nova::gputils
