#ifndef DEVICE_TRANSFER_INTERFACE_H
#define DEVICE_TRANSFER_INTERFACE_H

#include "device_resource_data.h"
#include "device_resource_descriptors.h"
#include "device_texture_descriptors.h"
/* Interface to load various nova resources to gpu , like textures , materials etc */

namespace device::gpgpu {

  bool validate_gpu_state();
  GPU_query_result ret_error();

  /* Allocate a buffer of size_bytes bytes on device*/
  GPU_query_result allocate_buffer(std::size_t buffer_size_bytes);

  /** copy_type :
   * 0 = host to device
   * 1 = device to host
   * 2 = device to device
   */
  GPU_query_result copy_buffer(const void *src, void *dest, std::size_t buffer_size_bytes, int copy_type);
  GPU_query_result deallocate_buffer(void *device_ptr);

  /**
   * if resc_desc has a null resource (for ex , null array but with RESOURCE_ARRAY type), this function will create the array and assign it to the
   * required resource_buffer_descriptor field.
   */
  GPU_texture create_texture(const void *src, int width, int height, const texture_descriptor &tex_desc, resource_descriptor &resc_desc);
  void destroy_texture(GPU_texture &texture);

  GPU_resource create_array(int width, int height, const channel_format &format, int flag);
  void destroy_array(GPU_resource &resource);

}  // namespace device::gpgpu

#endif  // DEVICE_TRANSFER_INTERFACE_H
