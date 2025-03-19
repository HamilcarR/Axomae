#include "DeviceImageTracker.h"
#include <internal/device/gpgpu/device_transfer_interface.h>
namespace nova::gpu {

  template<>
  void DeviceImageTracker<device::gpgpu::GPUTexture>::setDefaultDescriptors() {
    texture_descriptor.address_mode[0] = device::gpgpu::ADDRESS_WRAP;
    texture_descriptor.address_mode[1] = device::gpgpu::ADDRESS_WRAP;
    texture_descriptor.filter_mode = device::gpgpu::FILTER_POINT;
    texture_descriptor.normalized_coords = true;
    texture_descriptor.read_mode = device::gpgpu::READ_ELEMENT_TYPE;
  }

  template<>
  DeviceImageTracker<device::gpgpu::GPUTexture>::DeviceImageTracker(GLuint image_id, GLenum texture_type, device::gpgpu::ACCESS_TYPE access_mode)
      : DeviceMemoryTracker(image_id) {
    setDefaultDescriptors();
    DEVICE_ERROR_CHECK(device::gpgpu::interop_register_glimage(gl_id, texture_type, gpgpu_api_handle, access_mode).error_status);
  }

  template<>
  DeviceImageTracker<device::gpgpu::GPUTexture>::DeviceImageTracker(GLuint image_id,
                                         GLenum texture_type,
                                         device::gpgpu::ACCESS_TYPE access_mode,
                                         const device::gpgpu::texture_descriptor &descriptor)
      : DeviceMemoryTracker(image_id) {
    texture_descriptor = descriptor;
    DEVICE_ERROR_CHECK(device::gpgpu::interop_register_glimage(gl_id, texture_type, gpgpu_api_handle, access_mode).error_status);
  }

  template<>
  void DeviceImageTracker<device::gpgpu::GPUTexture>::mapBuffer() {
    device::gpgpu::GPUArray gpu_array;
    auto query_result = device::gpgpu::interop_get_mapped_array(gpu_array, gpgpu_api_handle, 0, 0);
    DEVICE_ERROR_CHECK(query_result.error_status);
    if (query_result.error_status.isValid()) {
      gpu_texture = device::gpgpu::GPUTexture(std::move(gpu_array), texture_descriptor);
    }
  }

}  // namespace nova::gpu