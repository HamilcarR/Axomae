#ifndef TEXTURE_DATASTRUCTURES_H
#define TEXTURE_DATASTRUCTURES_H
#include <glm/glm.hpp>
#include <internal/common/axstd/span.h>
#include <internal/device/gpgpu/device_texture_descriptors.h>

namespace nova::texturing {
  class TextureCtx;

  struct pipelined_geometric_data_s {
    glm::vec3 sampling_vector{};
  };
  struct texture_data_aggregate_s {
    pipelined_geometric_data_s geometric_data;
    const TextureCtx *texture_ctx{nullptr};
  };

  template<class T>
  struct TextureRawData {
    axstd::span<const T> raw_data;
    int width;
    int height;
    int channels;
  };
}  // namespace nova::texturing

using F32Texture = nova::texturing::TextureRawData<float>;
using F64Texture = nova::texturing::TextureRawData<double>;
using U32Texture = nova::texturing::TextureRawData<uint32_t>;

using U32ImgTexView = axstd::span<U32Texture>;
using CstU32ImgTexView = axstd::span<const U32Texture>;
using F32ImgTexView = axstd::span<F32Texture>;
using CstF32ImgTexView = axstd::span<const F32Texture>;
using IntopImgTexView = axstd::span<const device::gpgpu::APITextureHandle>;

namespace nova::texturing {
  struct u32tex_shared_views_s {
    CstU32ImgTexView u32_host;
    CstU32ImgTexView u32_device;
    IntopImgTexView interop_handles;
  };
  struct f32tex_shared_views_s {
    CstF32ImgTexView f32_host;
    CstF32ImgTexView f32_device;
    IntopImgTexView interop_handles;
  };
}  // namespace nova::texturing

#endif  // TEXTURE_DATASTRUCTURES_H
