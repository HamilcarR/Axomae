#include "nova_gpu_utils.h"
#include "internal/device/gpgpu/device_transfer_interface.h"

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

#else
  void lock_host_memory_default(const device_shared_caches_t &collection) { EMPTY_FUNCBODY }
  void unlock_host_memory(const device_shared_caches_t &collection) { EMPTY_FUNCBODY }

#endif
}  // namespace nova::gputils