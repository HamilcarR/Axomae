#ifndef MESH_DEVICE_RESOURCES_H
#define MESH_DEVICE_RESOURCES_H

#include <internal/common/axstd/span.h>
#include <internal/debug/Logger.h>
#include <internal/device/gpgpu/device_transfer_interface.h>
#include <internal/macro/project_macros.h>

namespace nova::gpu {

  /** Represents a buffer located in gpu memory .
   * @tparam T type of the buffer's elements.
   */
  template<class T>
  class DeviceBufferTracker {
    using handle_type = device::gpgpu::GPUGraphicsResrcHandle;
    using buffer_type = T;
    using stream_type = device::gpgpu::GPUStream;

   private:
    axstd::span<buffer_type> device_buffer{};
    handle_type gpgpu_api_handle{};
    GLuint vbo_id{};

   public:
    DeviceBufferTracker() = default;

    DeviceBufferTracker(uint32_t vbo_id_, device::gpgpu::ACCESS_TYPE access_mode) : vbo_id(vbo_id_) {
      DEVICE_ERROR_CHECK(device::gpgpu::interop_register_glbuffer(vbo_id, gpgpu_api_handle, access_mode).error_status);
    }

    ax_no_discard bool isValid() const { return !device_buffer.empty() && gpgpu_api_handle.isRegistered() && vbo_id != 0; }

    ~DeviceBufferTracker() {
      if (isValid())
        DEVICE_ERROR_CHECK(device::gpgpu::interop_unregister_resrc(gpgpu_api_handle).error_status);
      device_buffer = axstd::span<buffer_type>();
      vbo_id = 0;
    }

    DeviceBufferTracker(const DeviceBufferTracker &) = delete;
    DeviceBufferTracker &operator=(const DeviceBufferTracker &) = delete;
    DeviceBufferTracker(DeviceBufferTracker &&other) noexcept
        : device_buffer(std::move(other.device_buffer)), gpgpu_api_handle(std::move(other.gpgpu_api_handle)) {
      other.device_buffer = axstd::span<T>(nullptr, 0);
      vbo_id = other.vbo_id;
      other.vbo_id = 0;
    }

    DeviceBufferTracker &operator=(DeviceBufferTracker &&other) noexcept {
      device_buffer = other.device_buffer;
      other.device_buffer = axstd::span<T>(nullptr, 0);
      gpgpu_api_handle = std::move(other.gpgpu_api_handle);
      vbo_id = other.vbo_id;
      other.vbo_id = 0;
      return *this;
    }

    void mapResource() {
      device::gpgpu::GPUStream current_stream;
      auto query_result = device::gpgpu::interop_map_resrc(1, &gpgpu_api_handle, current_stream);
      DEVICE_ERROR_CHECK(query_result.error_status);
    }

    void unmapResource() {
      if (!gpgpu_api_handle.isMapped())
        return;
      device::gpgpu::GPUStream current_stream;
      auto query_result = device::gpgpu::interop_unmap_resrc(1, &gpgpu_api_handle, current_stream);
      DEVICE_ERROR_CHECK(query_result.error_status);
    }

    void mapResource(stream_type &stream) {
      auto query_result = device::gpgpu::interop_map_resrc(1, &gpgpu_api_handle, stream);
      DEVICE_ERROR_CHECK(query_result.error_status);
    }

    void unmapResource(stream_type &stream) {
      if (!gpgpu_api_handle.isMapped())
        return;
      auto query_result = device::gpgpu::interop_unmap_resrc(1, &gpgpu_api_handle, stream);
      DEVICE_ERROR_CHECK(query_result.error_status);
    }

    /* Caller of this method needs to call mapResource() beforehand.*/
    void mapBuffer() {
      auto query_result = device::gpgpu::interop_get_mapped_ptr(gpgpu_api_handle);
      DEVICE_ERROR_CHECK(query_result.error_status);
      if (!query_result.device_ptr) {
        LOG("GPU resource handle returned null for vbo id :" + std::to_string(vbo_id), LogLevel::ERROR);
        return;
      }
      device_buffer = axstd::span(static_cast<buffer_type *>(query_result.device_ptr), query_result.size / sizeof(buffer_type));
    }

    const axstd::span<buffer_type> &getDeviceBuffer() const { return device_buffer; }
    axstd::span<buffer_type> &getDeviceBuffer() { return device_buffer; }
  };

}  // namespace nova::gpu
#endif