#ifndef DEVICEIMAGETRACKER_H
#define DEVICEIMAGETRACKER_H

#include "DeviceMemoryTracker.h"

namespace nova::gpu {
  template<class T>
  class DeviceImageTracker final : public DeviceMemoryTracker {
   public:
    using handle_type = device::gpgpu::GPUGraphicsResrcHandle;
    using stream_type = device::gpgpu::GPUStream;
    using gpu_texture_type = T;
   private:
    gpu_texture_type gpu_texture;
    device::gpgpu::texture_descriptor texture_descriptor;

    void setDefaultDescriptors();

   public:
    DeviceImageTracker() = default;
    DeviceImageTracker(GLuint image_id, GLenum texture_type, device::gpgpu::ACCESS_TYPE access_mode);
    DeviceImageTracker(GLuint image_id,
                       GLenum texture_type,
                       device::gpgpu::ACCESS_TYPE access_mode,
                       const device::gpgpu::texture_descriptor &descriptor);
    ~DeviceImageTracker() override = default;
    DeviceImageTracker(const DeviceImageTracker &other) = delete;
    DeviceImageTracker(DeviceImageTracker &&other) noexcept = default;
    DeviceImageTracker &operator=(const DeviceImageTracker &other) = delete;
    DeviceImageTracker &operator=(DeviceImageTracker &&other) noexcept = default;

    device::gpgpu::APITextureHandle getImageID() const { return gpu_texture.id(); }
    /* Only takes GL_TEXTURE_2D. */
    void mapBuffer() override;
  };
}  // namespace nova::gpu

#endif
