#include "nova_gpu_utils.h"
#include "internal/device/gpgpu/device_transfer_interface.h"
#include "internal/device/gpgpu/device_utils.h"
#include "internal/memory/MemoryArena.h"

namespace resrc = device::gpgpu;
namespace nova::gputils {
#if defined(AXOMAE_USE_CUDA)
  void lock_host_memory_default(const device_shared_caches_t &collection) {
    for (const auto &element : collection.contiguous_caches)
      DEVICE_ERROR_CHECK(resrc::pin_host_memory(element.data(), element.size(), device::gpgpu::PIN_MODE_DEFAULT).error_status);
  }

  void unlock_host_memory(const device_shared_caches_t &collection) {
    for (const auto &element : collection.contiguous_caches)
      DEVICE_ERROR_CHECK(resrc::unpin_host_memory(element.data()).error_status);
  }

  gpu_random_generator_t initialize_rand(const kernel_argpack_t &argpack, core::memory::ByteArena &arena) {
    gpu_random_generator_t gpu_generators;
    gpu_generators.sobol.init(argpack, arena);
    gpu_generators.xorshift.init(argpack);
    return gpu_generators;
  }

  void clean_generators(gpu_random_generator_t &generators, core::memory::ByteArena &arena) {
    generators.sobol.cleanStates(arena);
    generators.xorshift.cleanStates();
  }

  gpu_util_structures_t initialize_gpu_structures(unsigned thread_distrib, core::memory::ByteArena &arena) {
    kernel_argpack_t argpack;
    if (thread_distrib % AX_GPU_WARP_SIZE != 0) {
      thread_distrib = (thread_distrib + (AX_GPU_WARP_SIZE - 1)) & ~(AX_GPU_WARP_SIZE - 1);
    }

    argpack.block_size = AX_GPU_WARP_SIZE;
    argpack.num_blocks = (thread_distrib + argpack.block_size.x - 1) / argpack.block_size.x;

    gpu_util_structures_t gpu_structures;
    gpu_structures.threads_distribution = argpack;
    gpu_structures.random_generator = initialize_rand(argpack, arena);
    return gpu_structures;
  }

  void cleanup_gpu_structures(gpu_util_structures_t &gpu_structures, core::memory::ByteArena &arena) {
    gpu_structures.random_generator.sobol.cleanStates(arena);
    gpu_structures.random_generator.xorshift.cleanStates();
  }

#else
  void lock_host_memory_default(const device_shared_caches_t &collection) { EMPTY_FUNCBODY }
  void unlock_host_memory(const device_shared_caches_t &collection){EMPTY_FUNCBODY}

  gpu_random_generator_t initialize_rand(const kernel_argpack_t &argpack) {
    return {};
  }
  void clean_generators(gpu_random_generator_t &generators) { EMPTY_FUNCBODY }

  void cleanup_gpu_structures(gpu_structures_t &gpu_structures) { EMPTY_FUNCBODY }

#endif
}  // namespace nova::gputils