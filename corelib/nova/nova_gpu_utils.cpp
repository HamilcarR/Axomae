#include "nova_gpu_utils.h"
#include "internal/device/gpgpu/device_transfer_interface.h"
#include "internal/device/gpgpu/device_utils.h"
#include "internal/memory/MemoryArena.h"

namespace resrc = device::gpgpu;
namespace nova::gputils {
#if defined(AXOMAE_USE_CUDA)
  void lock_host_memory_default(device_shared_caches_t &collection) {
    for (auto &element : collection.contiguous_caches)
      if (!element.locked){
        DEVICE_ERROR_CHECK(resrc::pin_host_memory(element.buffer.data(), element.buffer.size(), device::gpgpu::PIN_MODE_DEFAULT).error_status);
        element.locked = true;
      }
  }

  void unlock_host_memory( device_shared_caches_t &collection) {
    for (auto &element : collection.contiguous_caches)
      if (element.locked){
        DEVICE_ERROR_CHECK(resrc::unpin_host_memory(element.buffer.data()).error_status);
        element.locked = false;
      }
  }

  static gpu_random_generator_t initialize_rand(const kernel_argpack_t &argpack) {
    gpu_random_generator_t gpu_generators;
    gpu_generators.sobol.init(argpack, 7);
    gpu_generators.xorshift.init(argpack);
    return gpu_generators;
  }

  void clean_generators(gpu_random_generator_t &generators) {
    generators.sobol.cleanStates();
    generators.xorshift.cleanStates();
  }

  void initialize_gpu_structures(const domain2d &domain, gpu_util_structures_t &gpu_structures) {
    kernel_argpack_t argpack;
    argpack.block_size = {AX_GPU_WARP_SIZE, AX_GPU_WARP_SIZE};
    unsigned block_x = std::ceil((domain.x + argpack.block_size.x - 1) / argpack.block_size.x);
    unsigned block_y = std::ceil((domain.y + argpack.block_size.y - 1) / argpack.block_size.y);
    argpack.num_blocks = {block_x, block_y};

    AX_ASSERT_LT(argpack.num_blocks, AX_GPU_MAX_BLOCKDIM);
    gpu_structures.threads_distribution = argpack;
    gpu_structures.random_generator = initialize_rand(argpack);
  }

  void cleanup_gpu_structures(gpu_util_structures_t &gpu_structures) {
    gpu_structures.random_generator.sobol.cleanStates();
    gpu_structures.random_generator.xorshift.cleanStates();
  }

#else
  void lock_host_memory_default(device_shared_caches_t &) { EMPTY_FUNCBODY }
  void unlock_host_memory(device_shared_caches_t &) { EMPTY_FUNCBODY }

  void clean_generators(gpu_random_generator_t &) { EMPTY_FUNCBODY }

  void cleanup_gpu_structures(gpu_util_structures_t &){EMPTY_FUNCBODY}

  void initialize_gpu_structures(const domain2d & , gpu_util_structures_t& ) {EMPTY_FUNCBODY}

#endif
}  // namespace nova::gputils