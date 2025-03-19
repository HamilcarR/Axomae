#include "private/utils.h"

#include <cuda_runtime_api.h>
#include <internal/debug/Logger.h>
#include <internal/device/gpgpu/device_texture_descriptors.h>

namespace device::gpgpu {

  GPUTexture::GPUTexture(GPUArray &&array, const texture_descriptor &desc) : texture_array(std::move(array)) {
    cudaResourceDesc resc{};
    resc.resType = cudaResourceTypeArray;
    resc.res.array.array = retrieve_internal(texture_array);

    cudaTextureDesc tex_desc{};
    tex_desc.addressMode[0] = get_address_mode(desc.address_mode[0]);
    tex_desc.addressMode[1] = get_address_mode(desc.address_mode[1]);
    tex_desc.filterMode = get_filter_mode(desc);
    tex_desc.readMode = get_read_mode(desc);
    tex_desc.normalizedCoords = desc.normalized_coords;
    DeviceError err = cudaCreateTextureObject(&texture_object, &resc, &tex_desc, nullptr);
    DEVICE_ERROR_CHECK(err);
    LOG("Creating cuda texture id: " + std::to_string(texture_object), LogLevel::INFO);
  }
  const APITextureHandle &GPUTexture::id() const { return texture_object; }

  GPUTexture::~GPUTexture() {
    if (texture_object != 0) {
      LOG("Destroying cuda texture id: " + std::to_string(texture_object), LogLevel::INFO);
      DeviceError err = cudaDestroyTextureObject(texture_object);
      texture_object = 0;
      DEVICE_ERROR_CHECK(err);
    }
  }

  GPUTexture::GPUTexture(GPUTexture &&other) noexcept {
    texture_object = other.texture_object;
    other.texture_object = 0;
    texture_array = std::move(other.texture_array);
  }

  GPUTexture &GPUTexture::operator=(GPUTexture &&other) noexcept {
    texture_object = other.texture_object;
    other.texture_object = 0;
    texture_array = std::move(other.texture_array);
    return *this;
  }

}  // namespace device::gpgpu