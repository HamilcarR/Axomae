#ifndef DEVICEIMAGETRACKER_H
#define DEVICEIMAGETRACKER_H

#include "DeviceMemoryTracker.h"

namespace nova::gpu {
  template<class T>
  class DeviceImageTracker : public DeviceMemoryTracker {
   public:
    using handle_type = device::gpgpu::GPUGraphicsResrcHandle;
    using stream_type = device::gpgpu::GPUStream;
    using gpu_texture_type = T;

   private:
    gpu_texture_type gpu_texture;
    device::gpgpu::texture_descriptor texture_descriptor;

    void setDefaultDescriptors() {
      texture_descriptor.address_mode[0] = device::gpgpu::ADDRESS_WRAP;
      texture_descriptor.address_mode[1] = device::gpgpu::ADDRESS_WRAP;
      texture_descriptor.filter_mode = device::gpgpu::FILTER_POINT;
      texture_descriptor.normalized_coords = true;
      texture_descriptor.read_mode = device::gpgpu::READ_ELEMENT_TYPE;
    }

   public:
    DeviceImageTracker() = default;
    DeviceImageTracker(GLuint image_id, GLenum texture_type, device::gpgpu::ACCESS_TYPE access_mode) : DeviceMemoryTracker(image_id) {
      setDefaultDescriptors();
      DEVICE_ERROR_CHECK(device::gpgpu::interop_register_glimage(gl_id, texture_type, gpgpu_api_handle, access_mode).error_status);
    }

    DeviceImageTracker(GLuint image_id,
                       GLenum texture_type,
                       device::gpgpu::ACCESS_TYPE access_mode,
                       const device::gpgpu::texture_descriptor &descriptor)
        : DeviceMemoryTracker(image_id) {
      texture_descriptor = descriptor;
      DEVICE_ERROR_CHECK(device::gpgpu::interop_register_glimage(gl_id, texture_type, gpgpu_api_handle, access_mode).error_status);
    }

    ~DeviceImageTracker() override = default;
    DeviceImageTracker(const DeviceImageTracker &other) = delete;
    DeviceImageTracker(DeviceImageTracker &&other) noexcept = default;
    DeviceImageTracker &operator=(const DeviceImageTracker &other) = delete;
    DeviceImageTracker &operator=(DeviceImageTracker &&other) noexcept = default;

    device::gpgpu::APITextureHandle getImageID() const { return gpu_texture.id(); }
    /* Only takes GL_TEXTURE_2D. */
    void mapBuffer() override {
      device::gpgpu::GPUArray gpu_array;
      auto query_result = device::gpgpu::interop_get_mapped_array(gpu_array, gpgpu_api_handle, 0, 0);
      DEVICE_ERROR_CHECK(query_result.error_status);
      if (query_result.error_status.isValid()) {
        gpu_texture = gpu_texture_type(std::move(gpu_array), texture_descriptor);
      }
    }
  };
}  // namespace nova::gpu

#endif
