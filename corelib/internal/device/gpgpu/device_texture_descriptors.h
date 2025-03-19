#ifndef DEVICE_TEXTURE_DESCRIPTORS_H
#define DEVICE_TEXTURE_DESCRIPTORS_H

#include "device_resource_descriptors.h"
#include <any>
#ifdef AXOMAE_USE_CUDA
#  include <texture_types.h>
#endif
namespace device::gpgpu {

  enum ADDRESS_MODE { ADDRESS_WRAP = 0x00, ADDRESS_CLAMP = 0x01, ADDRESS_MIRROR = 0x02, ADDRESS_BORDER = 0x03 };

  /**
   * FILTER_POINT = No interpolation when fetching texel
   * FILTER_LINEAR = Performs linear interpolation
   */
  enum FILTER_MODE { FILTER_POINT = 0x00, FILTER_LINEAR = 0x01 };

  /**
   * READ_ELEMENT_TYPE = fetch texel in whatever type it is stored
   * READ_NORMALIZED_FLOAT = divides texel value by it's max type to be in 0.f,1.f range
   */
  enum READ_MODE { READ_ELEMENT_TYPE = 0x00, READ_NORMALIZED_FLOAT = 0x01 };
  struct texture_descriptor {
    channel_format channel_descriptor;
    ADDRESS_MODE address_mode[3]{};
    READ_MODE read_mode;
    FILTER_MODE filter_mode;
    bool normalized_coords;
    texture_descriptor() = default;

    texture_descriptor(const texture_descriptor &other) {
      channel_descriptor = other.channel_descriptor;
      address_mode[0] = other.address_mode[0];
      address_mode[1] = other.address_mode[1];
      address_mode[2] = other.address_mode[2];
      read_mode = other.read_mode;
      filter_mode = other.filter_mode;
      normalized_coords = other.normalized_coords;
    }

    texture_descriptor &operator=(const texture_descriptor &other) {
      channel_descriptor = other.channel_descriptor;
      address_mode[0] = other.address_mode[0];
      address_mode[1] = other.address_mode[1];
      address_mode[2] = other.address_mode[2];
      read_mode = other.read_mode;
      filter_mode = other.filter_mode;
      normalized_coords = other.normalized_coords;
      return *this;
    }
  };

  // TODO : need to change this asap !
  struct GPU_texture {
    std::any texture_object;
    std::any array_object;
    channel_format descriptor;
  };

  /************************************************************************************************************************************************************************************/

#ifdef AXOMAE_USE_CUDA
  using APITextureHandle = cudaTextureObject_t;
#elif defined(AXOMAE_USE_OPENCL)
  using APITextureHandle = cl_mem;
#elif defined(AXOMAE_USE_HIP)
  using APITextureHandle = hipTextureObject_t;
#else
  using APITextureHandle = uint32_t;
#endif

  class GPUTexture {
    APITextureHandle texture_object{0};
    GPUArray texture_array;

   public:
    GPUTexture() = default;
    GPUTexture(GPUArray &&array, const texture_descriptor &tex_desc);
    ~GPUTexture();
    GPUTexture(const GPUTexture &other) = delete;
    GPUTexture &operator=(const GPUTexture &other) = delete;
    GPUTexture &operator=(GPUTexture &&other) noexcept;
    GPUTexture(GPUTexture &&other) noexcept;
    const APITextureHandle &id() const;
  };

}  // namespace device::gpgpu

#endif  // DEVICE_TEXTURE_DESCRIPTORS_H
