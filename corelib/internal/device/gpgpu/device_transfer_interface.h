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
  /*******************************************************************************************************************************************************************************/
  /** Context Sharing **/
  void init_driver_API();
  void create_context(GPUContext &context);
  void set_current_context(GPUContext &context);
  /*******************************************************************************************************************************************************************************/
  /** Device Synchronizations **/
  GPU_query_result synchronize_device();

  /*******************************************************************************************************************************************************************************/
  /** Streams **/
  GPU_query_result create_stream(GPUStream &stream);

  /*******************************************************************************************************************************************************************************/
  /** Generic buffers allocation and deallocation **/

  enum COPY_MODE { HOST_HOST = 0, HOST_DEVICE = 1, DEVICE_HOST = 2, DEVICE_DEVICE = 3 };
  GPU_query_result allocate_device_managed(std::size_t buffer_size_bytes, bool global_access);
  GPU_query_result allocate_buffer(std::size_t buffer_size_bytes);
  GPU_query_result allocate_symbol(void **symbol, std::size_t buffer_size_bytes);
  GPU_query_result copy_buffer(const void *src, void *dest, std::size_t buffer_size_bytes, COPY_MODE copy_type);
  GPU_query_result copy_to_symbol(const void *src, void *dest, std::size_t buffer_size_bytes, COPY_MODE copy_type);
  GPU_query_result deallocate_buffer(void *device_ptr);

  /*******************************************************************************************************************************************************************************/
  /** Textures **/

  /**
   * if resc_desc has a null resource (for ex , null array but with RESOURCE_ARRAY type), this function will create the array and assign it to the
   * required resource_buffer_descriptor field :
   * cudaArray for array , cudaMipmappedArray for mipmap etc.
   */
  GPU_texture create_texture(const void *src, int width, int height, const texture_descriptor &tex_desc, resource_descriptor &resc_desc);
  void destroy_texture(GPU_texture &texture);
  GPU_resource create_array(int width, int height, const channel_format &format, int flag);
  GPU_query_result destroy_array(GPU_resource &resource);

  /*******************************************************************************************************************************************************************************/
  /** Unified memory access */

  enum PIN_EXT { PIN_EXT_NOOP };
  enum PIN_MODE { PIN_MODE_DEFAULT = 0, PIN_MODE_PORTABLE = 1, PIN_MODE_MAPPED = 2, PIN_MODE_IO = 3, PIN_MODE_RO = 4 };

  /************************************************************************************************************************************************************************************/
  /**
   * Representation of a buffer in memory. Can be a GPU memory or CPU memory buffer.
   */
  class DeviceSharedBufferView {
    class Impl;
    std::shared_ptr<Impl> pimpl;

   public:
    DeviceSharedBufferView();
    /**
     * Creates a GPUDeviceBuffer .
     * @tparam T Type of the buffer's elements.
     * @param buffer Buffer memory to reference.
     * @param size Number of T elements.
     */
    template<class T>
    DeviceSharedBufferView(T *buffer, unsigned size) {
      initBuffer(buffer, size * sizeof(T));
    }

    ~DeviceSharedBufferView();
    DeviceSharedBufferView(const DeviceSharedBufferView &) = delete;
    DeviceSharedBufferView &operator=(const DeviceSharedBufferView &) = delete;
    DeviceSharedBufferView(DeviceSharedBufferView &&) noexcept;
    DeviceSharedBufferView &operator=(DeviceSharedBufferView &&) noexcept;

    /**
     * Returns size of buffer in number of elements.
     * @tparam T Type of the element of the buffer.
     * @return Total size of the underlying buffer in number of elements.
     */
    template<class T>
    std::size_t size() const {
      return bufferSizeBytes() / sizeof(T);
    }

    /**
     * Returns an address to the underlying buffer and casts it to T
     * @tparam T Type of the returned buffer.
     * @return Address of the buffer.
     */
    template<class T>
    T *data() const {
      return static_cast<T *>(getCastData());
    }

    /**
     * Checks if the buffer has been pinned.
     */
    bool isMapped() const;
    friend GPU_query_result pin_host_memory(DeviceSharedBufferView &buffer, PIN_MODE mode);
    friend GPU_query_result unpin_host_memory(DeviceSharedBufferView &buffer);

   private:
    void initBuffer(void *buffer, size_t size);
    void initBuffer(const void *buffer, size_t size);
    void *getCastData() const;
    std::size_t bufferSizeBytes() const;
  };

  template<>
  std::size_t DeviceSharedBufferView::size<void>() const;

  GPU_query_result pin_host_memory(void *buffer, std::size_t host_buffer_in_bytes, PIN_MODE mode);
  GPU_query_result unpin_host_memory(void *buffer);
  GPU_query_result get_pinned_memory_dptr(void *host_ptr, PIN_EXT flag = PIN_EXT_NOOP);

  /*******************************************************************************************************************************************************************************/
  /** OpenGL<-> CUDA interop **/

  /**
   * Initializes a GPUGraphicsResrcHandle from a GL vbo.
   */
  GPU_query_result interop_register_glbuffer(GLuint vbo_id, GPUGraphicsResrcHandle &resource_handle, ACCESS_TYPE access_type);
  GPU_query_result interop_register_glimage(GLuint tex_id, GLenum target, GPUGraphicsResrcHandle &resource_handle, ACCESS_TYPE access_type);
  GPU_query_result interop_unregister_resrc(GPUGraphicsResrcHandle &gpu_graphics_resource);
  GPU_query_result interop_map_resrc(int count, GPUGraphicsResrcHandle *gpu_resources_array, GPUStream &stream);
  GPU_query_result interop_unmap_resrc(int count, GPUGraphicsResrcHandle *gpu_resources_array, GPUStream &stream);
  GPU_query_result interop_get_mapped_ptr(GPUGraphicsResrcHandle &gpu_graphics_resource);

}  // namespace device::gpgpu

#endif  // DEVICE_TRANSFER_INTERFACE_H
