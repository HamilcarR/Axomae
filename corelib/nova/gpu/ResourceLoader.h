#ifndef UTILS_H
#define UTILS_H

#include "DeviceError.h"
#include "manager/NovaExceptionManager.h"
#include "project_macros.h"

/* Interface to load various nova resources to gpu , like textures , materials etc */

namespace nova::gpu::resrc {

  struct GPUResource {
    void *device_ptr{};
    DeviceError error_status;
  };

  bool validate_gpu_state();
  GPUResource ret_error();

  /* Allocate a buffer of size_bytes bytes on device*/
  GPUResource allocate(std::size_t buffer_size_bytes);

  /* copy_type :
   * 0 = host to device
   * 1 = device to host
   * 2 = device to device
   */
  GPUResource copy(const void *src, void *dest, std::size_t buffer_size_bytes, int copy_type);

  GPUResource deallocate(void *device_ptr);

}  // namespace nova::gpu::resrc

#endif  // UTILS_H
