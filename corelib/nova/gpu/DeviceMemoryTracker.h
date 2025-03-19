#ifndef DEVICEMEMORYTRACKER_H
#define DEVICEMEMORYTRACKER_H

#include <interfaces/DeviceMemoryResourceTrackerInterface.h>
#include <internal/device/gpgpu/device_transfer_interface.h>

namespace nova::gpu {
  class DeviceMemoryTracker : public DeviceMemoryResourceTrackerInterface {
    using handle_type = device::gpgpu::GPUGraphicsResrcHandle;
    using stream_type = device::gpgpu::GPUStream;

   protected:
    handle_type gpgpu_api_handle{};
    GLuint gl_id{};

    DeviceMemoryTracker() = default;
    explicit DeviceMemoryTracker(GLuint id) : gl_id(id) {}

   public:
    ~DeviceMemoryTracker() override {
      if (DeviceMemoryTracker::isValid())
        DEVICE_ERROR_CHECK(device::gpgpu::interop_unregister_resrc(gpgpu_api_handle).error_status);
      gl_id = 0;
    }
    DeviceMemoryTracker(const DeviceMemoryTracker &other) = delete;

    DeviceMemoryTracker(DeviceMemoryTracker &&other) noexcept : gpgpu_api_handle(std::move(other.gpgpu_api_handle)), gl_id(other.gl_id) {
      other.gl_id = 0;
    }

    DeviceMemoryTracker &operator=(const DeviceMemoryTracker &other) = delete;

    DeviceMemoryTracker &operator=(DeviceMemoryTracker &&other) noexcept {
      gpgpu_api_handle = std::move(other.gpgpu_api_handle);
      gl_id = other.gl_id;
      other.gl_id = 0;
      return *this;
    }

    void mapResource() override {
      device::gpgpu::GPUStream current_stream;
      auto query_result = device::gpgpu::interop_map_resrc(1, &gpgpu_api_handle, current_stream);
      DEVICE_ERROR_CHECK(query_result.error_status);
    }

    void unmapResource() override {
      if (!gpgpu_api_handle.isMapped())
        return;
      device::gpgpu::GPUStream current_stream;
      auto query_result = device::gpgpu::interop_unmap_resrc(1, &gpgpu_api_handle, current_stream);
      DEVICE_ERROR_CHECK(query_result.error_status);
    }

    void mapResource(stream_type &stream) override {
      auto query_result = device::gpgpu::interop_map_resrc(1, &gpgpu_api_handle, stream);
      DEVICE_ERROR_CHECK(query_result.error_status);
    }

    void unmapResource(stream_type &stream) override {
      if (!gpgpu_api_handle.isMapped())
        return;
      auto query_result = device::gpgpu::interop_unmap_resrc(1, &gpgpu_api_handle, stream);
      DEVICE_ERROR_CHECK(query_result.error_status);
    }

    ax_no_discard bool isValid() const override { return gpgpu_api_handle.isRegistered() && gl_id != 0; }
  };

}  // namespace nova::gpu
#endif  // DEVICEMEMORYTRACKER_H
