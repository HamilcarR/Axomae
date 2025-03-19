#include "TextureContext.h"

#include <internal/common/math/math_texturing.h>
#include <internal/common/utils.h>
namespace nova::texturing {

  ax_device_callable const F32Texture &TextureBundleViews::getF32(std::size_t index) const {
#ifndef __CUDA_ARCH__
    return f32_textures.f32_host[index];
#else
    return f32_textures.f32_device[index];
#endif
  }

  ax_device_callable const device::gpgpu::APITextureHandle &TextureBundleViews::getDeviceTextureHandle(std::size_t index, FORMAT format) const {
    switch (format) {
      case FLOAT32:
        return f32_textures.interop_handles[index];
      default:
        return u32_textures.interop_handles[index];
    }
  }
  int TextureBundleViews::getChannelsU32(std::size_t index) const { return getU32(index).channels; }
  int TextureBundleViews::getChannelsF32(std::size_t index) const { return getF32(index).channels; }

  ax_device_callable const U32Texture &TextureBundleViews::getU32(std::size_t index) const {
#ifndef __CUDA_ARCH__
    return u32_textures.u32_host[index];
#else
    return u32_textures.u32_device[index];
#endif
  }

  ax_device_callable TextureBundleViews::TextureBundleViews(const u32tex_shared_views_s &utex, const f32tex_shared_views_s &ftex)
      : u32_textures(utex), f32_textures(ftex) {}

  ax_device_callable TextureBundleViews::TextureBundleViews(const u32tex_shared_views_s &utex) : u32_textures(utex) {}

  ax_device_callable const U32Texture &TextureCtx::u32texture(std::size_t index) const { return bundle.getU32(index); }

  ax_device_callable const F32Texture &TextureCtx::f32texture(std::size_t index) const { return bundle.getF32(index); }

  ax_device_callable TextureCtx::TextureCtx(const TextureBundleViews &b, bool use_itp) : bundle(b), use_interop(use_itp) {}

  ax_device_callable int TextureCtx::u32width(std::size_t index) const {
    const U32Texture tex = bundle.getU32(index);
    return tex.width;
  }

  ax_device_callable int TextureCtx::u32height(std::size_t index) const {
    const U32Texture tex = bundle.getU32(index);
    return tex.height;
  }

  ax_device_callable int TextureCtx::u32channels(std::size_t texture_index) const { return bundle.getChannelsU32(texture_index); }

  /* Should be replaced with a solution that handles out of range UV mapping. */
  ax_device_callable_inlined unsigned uv2index(float t, int dim) {
    float a = AX_GPU_ABS(t);
    float rem = a - AX_GPU_FLOORF(a);
    unsigned i = math::texture::uvToPixel(rem, dim - 1);
    return i;
  }

  ax_device_callable uint32_t TextureCtx::u32pixel(std::size_t texture_index, float u, float v) const {
    if constexpr (core::build::is_gpu_build) {
      if (use_interop) {
#ifdef __CUDA_ARCH__
        device::gpgpu::APITextureHandle sampler2D = bundle.getDeviceTextureHandle(texture_index, TextureBundleViews::UINT32);
        uchar4 pixel = tex2D<uchar4>(sampler2D, u, v);
        return pixel.z << 24 | pixel.y << 16 | pixel.x << 8 | pixel.w;
#endif
        //  Other backend texture sampling methods will in the future be implemented here.
      }
    }
    const int width = u32width(texture_index);
    const int height = u32height(texture_index);
    unsigned i = uv2index(u, width);
    unsigned j = uv2index(v, height);
    unsigned idx = (i * height + j);
    AX_ASSERT_LT(idx, width * height);
    return u32texture(texture_index).raw_data[idx];
  }
}  // namespace nova::texturing