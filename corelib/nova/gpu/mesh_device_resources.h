#ifndef MESH_DEVICE_RESOURCES_H
#define MESH_DEVICE_RESOURCES_H

#include "DeviceMemoryTracker.h"
#include <internal/common/axstd/span.h>
#include <internal/debug/Logger.h>
#include <internal/macro/project_macros.h>
namespace nova::gpu {

  /**
   * Represents a buffer located in gpu memory , shared between a cuda context and opengl.
   */
  template<class T>
  class DeviceBufferTracker final : public DeviceMemoryTracker {
   public:
    using handle_type = device::gpgpu::GPUGraphicsResrcHandle;
    using buffer_type = T;
    using stream_type = device::gpgpu::GPUStream;

   private:
    axstd::span<buffer_type> device_buffer{};

   public:
    DeviceBufferTracker() = default;

    ~DeviceBufferTracker() override { device_buffer = {}; }

    DeviceBufferTracker(GLuint vbo_id, device::gpgpu::ACCESS_TYPE access_mode) : DeviceMemoryTracker(vbo_id) {
      DEVICE_ERROR_CHECK(device::gpgpu::interop_register_glbuffer(gl_id, gpgpu_api_handle, access_mode).error_status);
    }

    /* Caller of this method needs to call mapResource() beforehand.*/
    void mapBuffer() override {
      auto query_result = device::gpgpu::interop_get_mapped_ptr(gpgpu_api_handle);
      DEVICE_ERROR_CHECK(query_result.error_status);
      if (!query_result.device_ptr) {
        LOG("GPU resource handle returned null for vbo id :" + std::to_string(gl_id), LogLevel::ERROR);
        return;
      }
      device_buffer = axstd::span(static_cast<buffer_type *>(query_result.device_ptr), query_result.size / sizeof(buffer_type));
    }

    DeviceBufferTracker(const DeviceBufferTracker &) = delete;

    DeviceBufferTracker &operator=(const DeviceBufferTracker &) = delete;

    DeviceBufferTracker(DeviceBufferTracker &&other) noexcept : DeviceMemoryTracker(std::move(other)), device_buffer(std::move(other.device_buffer)) {

      other.device_buffer = axstd::span<T>(nullptr, 0);
    }

    DeviceBufferTracker &operator=(DeviceBufferTracker &&other) noexcept {
      DeviceMemoryTracker::operator=(std::move(other));
      device_buffer = std::move(other.device_buffer);
      other.device_buffer = axstd::span<T>(nullptr, 0);
      return *this;
    }

    ax_no_discard bool isValid() const override { return !device_buffer.empty() && DeviceMemoryTracker::isValid(); }

    const axstd::span<buffer_type> &getDeviceBuffer() const { return device_buffer; }
    axstd::span<buffer_type> &getDeviceBuffer() { return device_buffer; }
  };

}  // namespace nova::gpu
#endif