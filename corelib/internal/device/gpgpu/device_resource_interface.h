#ifndef DEVICE_RESOURCE_INTERFACE_H
#define DEVICE_RESOURCE_INTERFACE_H

#include "device_resource_data.h"

/* Interface to load various nova resources to gpu , like textures , materials etc */

namespace device::gpgpu::resource {

  bool validate_gpu_state();
  GPU_resource ret_error();

  /* Allocate a buffer of size_bytes bytes on device*/
  GPU_resource allocate_buffer(std::size_t buffer_size_bytes);

  /* copy_type :
   * 0 = host to device
   * 1 = device to host
   * 2 = device to device
   */
  GPU_resource copy_buffer(const void *src, void *dest, std::size_t buffer_size_bytes, int copy_type);
  GPU_resource deallocate_buffer(void *device_ptr);

  GPU_texture create_texture(const void *src, int width, int height, const texture_channel_descriptor &desc);
  void destroy_texture(GPU_texture &texture);

}  // namespace device::gpgpu::resource

#endif  // DEVICE_RESOURCE_INTERFACE_H
