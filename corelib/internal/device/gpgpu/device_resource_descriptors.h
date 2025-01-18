#ifndef DEVICE_RESOURCE_DESCRIPTORS_H
#define DEVICE_RESOURCE_DESCRIPTORS_H

#include <any>
#include <memory>

namespace device::gpgpu {
  enum FORMAT_TYPE { FLOAT, UINT8X4N };

  enum ACCESS_TYPE { READ_WRITE, WRITE_ONLY, READ_ONLY };
  struct channel_format {
    int bits_size_x, bits_size_y, bits_size_z, bits_size_a;
    FORMAT_TYPE format_type;
  };

  enum RESOURCE_TYPE { RESOURCE_ARRAY = 0x00, RESOURCE_MIPMAP_ARRAY = 0x01, RESOURCE_LINEAR = 0x02, RESOURCE_PITCH2D = 0x03 };

  struct GPU_array {
    void *array;
  };

  struct GPU_mipmap {
    void *mipmap;
  };

  struct GPU_linear {
    void *device_ptr;
    channel_format format;
    std::size_t size_bytes;
  };

  struct GPU_pitch2d {
    void *device_ptr;
    channel_format format;
    size_t width;
    size_t height;
    size_t pitch_bytes;
  };

  struct GPU_resource {
    union {
      GPU_array array;
      GPU_mipmap mipmap;
      GPU_linear linear;
      GPU_pitch2d pitch2d;
    } res;
  };

  struct resource_descriptor {
    RESOURCE_TYPE type;
    GPU_resource resource_buffer_descriptors;
  };

  /************************************************************************************************************************************************************************************/
  class GPUGraphicsResrcHandle {
    class Impl;

   public:
    std::unique_ptr<Impl> pimpl;

    GPUGraphicsResrcHandle();
    ~GPUGraphicsResrcHandle();
    GPUGraphicsResrcHandle(const GPUGraphicsResrcHandle &) = delete;
    GPUGraphicsResrcHandle &operator=(const GPUGraphicsResrcHandle &) = delete;
    GPUGraphicsResrcHandle(GPUGraphicsResrcHandle &&) noexcept;
    GPUGraphicsResrcHandle &operator=(GPUGraphicsResrcHandle &&) noexcept;

    bool isMapped() const;
    bool isRegistered() const;
  };

  class GPUStream {
    class Impl;

   public:
    std::unique_ptr<Impl> pimpl;
    GPUStream();
    ~GPUStream();
    GPUStream(const GPUStream &) = delete;
    GPUStream &operator=(const GPUStream &) = delete;
    GPUStream(GPUStream &&) noexcept;
    GPUStream &operator=(GPUStream &&) noexcept;
  };

  /************************************************************************************************************************************************************************************/
  class GPUContext {
    class Impl;

   public:
    std::unique_ptr<Impl> pimpl;

    GPUContext();
    ~GPUContext();
    GPUContext(const GPUContext &) = delete;
    GPUContext &operator=(const GPUContext &) = delete;
    GPUContext(GPUContext &&) noexcept;
    GPUContext &operator=(GPUContext &&) noexcept;
  };

}  // namespace device::gpgpu
#endif
