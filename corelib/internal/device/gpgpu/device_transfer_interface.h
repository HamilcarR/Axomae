#ifndef DEVICE_TRANSFER_INTERFACE_H
#define DEVICE_TRANSFER_INTERFACE_H

#include "device_resource_data.h"
#include "device_resource_descriptors.h"
#include "device_texture_descriptors.h"
#include <internal/device/rendering/opengl/gl_headers.h>

/* Interface to load various resources to gpu in an agnostic way. */
namespace device::gpgpu {

  bool validate_gpu_state();
  GPU_query_result ret_error();
  GPU_query_result synchronize_device();

  /* Allocate a buffer of size_bytes bytes on device*/
  GPU_query_result allocate_buffer(std::size_t buffer_size_bytes);
  GPU_query_result allocate_symbol(void **symbol, std::size_t buffer_size_bytes);

  enum COPY_MODE { HOST_HOST = 0, HOST_DEVICE = 1, DEVICE_HOST = 2, DEVICE_DEVICE = 3 };
  GPU_query_result copy_buffer(const void *src, void *dest, std::size_t buffer_size_bytes, COPY_MODE copy_type);
  GPU_query_result copy_to_symbol(const void *src, void *dest, std::size_t buffer_size_bytes, COPY_MODE copy_type);
  GPU_query_result deallocate_buffer(void *device_ptr);

  /**
   * if resc_desc has a null resource (for ex , null array but with RESOURCE_ARRAY type), this function will create the array and assign it to the
   * required resource_buffer_descriptor field :
   * cudaArray for array , cudaMipmappedArray for mipmap etc.
   */
  GPU_texture create_texture(const void *src, int width, int height, const texture_descriptor &tex_desc, resource_descriptor &resc_desc);
  void destroy_texture(GPU_texture &texture);

  GPU_resource create_array(int width, int height, const channel_format &format, int flag);
  GPU_query_result destroy_array(GPU_resource &resource);

  enum PIN_MODE { PIN_MODE_DEFAULT = 0, PIN_MODE_PORTABLE = 1, PIN_MODE_MAPPED = 2, PIN_MODE_IO = 3, PIN_MODE_RO = 4 };
  GPU_query_result pin_host_memory(void *buffer, std::size_t host_buffer_in_bytes, PIN_MODE mode);
  GPU_query_result unpin_host_memory(void *buffer);
  enum PIN_EXT { PIN_EXT_NOOP };
  GPU_query_result get_pinned_memory_dptr(void *host_ptr, PIN_EXT flag = PIN_EXT_NOOP);

  /* interops */
  /* GL */
  /* Takes corresponding pointers depending on if we use cuda (or hip later).
   * In the case of cuda, we're working with a cudaGraphicsResource_t .
   */
  GPU_query_result interop_register_glbuffer(GLuint vbo_id, ACCESS_TYPE access_type);
  GPU_query_result interop_register_glimage(GLuint tex_id, GLenum target, ACCESS_TYPE access_type);
  GPU_query_result interop_unregister_resrc(void *gpu_graphics_resource);
  GPU_query_result interop_map_resrc(int count, void **gpu_resources_array, void *stream = nullptr);
  GPU_query_result interop_unmap_resrc(int count, void **gpu_resources_array, void *stream = nullptr);
  GPU_query_result interop_get_mapped_ptr(void *gpu_graphics_resource);

}  // namespace device::gpgpu

#endif  // DEVICE_TRANSFER_INTERFACE_H
