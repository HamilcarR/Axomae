#ifndef TEXTURECONTEXT_H
#define TEXTURECONTEXT_H
#include "texture_datastructures.h"
#include <internal/device/gpgpu/device_macros.h>

#include <internal/common/math/math_texturing.h>
#include <internal/common/utils.h>
#include <internal/device/gpgpu/device_utils.h>
#include <internal/macro/project_macros.h>

namespace nova::texturing {
  class TextureBundleViews {
   public:
    enum FORMAT { UINT32, FLOAT32 };

   private:
    u32tex_shared_views_s u32_textures;
    f32tex_shared_views_s f32_textures;

   public:
    CLASS_DCM(TextureBundleViews)

    ax_device_callable_inlined const device::gpgpu::APITextureHandle &getDeviceTextureHandle(std::size_t index, FORMAT format) const {
      switch (format) {
        case FLOAT32:
          return f32_textures.interop_handles[index];
        case UINT32:
          return u32_textures.interop_handles[index];
        default:
          AX_UNREACHABLE;
          return u32_textures.interop_handles[index];
      }
    }
    ax_device_callable_inlined int getChannelsU32(std::size_t index) const { return getU32(index).channels; }
    ax_device_callable_inlined int getChannelsF32(std::size_t index) const { return getF32(index).channels; }

    ax_device_callable_inlined const U32Texture &getU32(std::size_t index) const { return u32_textures.managed_tex_view[index]; }

    ax_device_callable_inlined TextureBundleViews(const u32tex_shared_views_s &utex, const f32tex_shared_views_s &ftex)
        : u32_textures(utex), f32_textures(ftex) {}

    ax_device_callable_inlined TextureBundleViews(const u32tex_shared_views_s &utex) : u32_textures(utex) {}

    ax_device_callable_inlined TextureBundleViews(const f32tex_shared_views_s &ftex) : f32_textures(ftex) {}

    ax_device_callable_inlined const F32Texture &getF32(std::size_t index) const { return f32_textures.managed_tex_view[index]; }
  };

  /* Should be replaced with a solution that handles out of range UV mapping. */
  ax_device_callable_inlined unsigned uv2index(float t, int dim) {
    float a = AX_GPU_ABS(t);
    float rem = a - AX_GPU_FLOORF(a);
    rem = rem == 0 ? a : rem;  // In case t is an integer so that rem will always be != 0 ;
    unsigned i = math::texture::uvToPixel(rem, dim);
    return i;
  }

  class TextureCtx {
    TextureBundleViews bundle;
    bool use_interop{true};

   public:
    CLASS_DCM(TextureCtx)

    ax_device_callable const TextureBundleViews &getBundle() const { return bundle; }

    ax_device_callable const U32Texture &u32texture(std::size_t index) const { return bundle.getU32(index); }

    ax_device_callable const F32Texture &f32texture(std::size_t index) const { return bundle.getF32(index); }

    ax_device_callable TextureCtx(const TextureBundleViews &b, bool use_itp = true) : bundle(b), use_interop(use_itp) {}

    ax_device_callable int u32width(std::size_t index) const {
      const U32Texture tex = bundle.getU32(index);
      return tex.width;
    }

    ax_device_callable int u32height(std::size_t index) const {
      const U32Texture tex = bundle.getU32(index);
      return tex.height;
    }

    ax_device_callable int f32height(std::size_t index) const {
      const F32Texture tex = bundle.getF32(index);
      return tex.height;
    }

    ax_device_callable int f32width(std::size_t index) const {
      const F32Texture tex = bundle.getF32(index);
      return tex.width;
    }

    ax_device_callable int f32channels(std::size_t index) const { return bundle.getChannelsF32(index); }

    ax_device_callable int u32channels(std::size_t index) const { return bundle.getChannelsU32(index); }

    ax_device_callable_inlined bool u32IsRGBA(std::size_t index) const {
      const U32Texture tex = bundle.getU32(index);
      return tex.is_rgba;
    }

    ax_device_callable_inlined bool f32IsRGBA(std::size_t index) const {
      const F32Texture tex = bundle.getF32(index);
      return tex.is_rgba;
    }

    ax_device_callable_inlined bool f32IsUInverted(std::size_t index) const {
      const F32Texture tex = bundle.getF32(index);
      return tex.invert_u;
    }

    ax_device_callable_inlined bool f32IsVInverted(std::size_t index) const {
      const F32Texture tex = bundle.getF32(index);
      return tex.invert_v;
    }

    ax_device_callable_inlined bool u32IsUInverted(std::size_t index) const {
      const U32Texture tex = bundle.getU32(index);
      return tex.invert_u;
    }

    ax_device_callable_inlined bool u32IsVInverted(std::size_t index) const {
      const U32Texture tex = bundle.getU32(index);
      return tex.invert_v;
    }

    /* Outputs pixel in BGRA format. */
    ax_device_callable uint32_t u32pixel(std::size_t texture_index, float u, float v) const {
      u = u32IsUInverted(texture_index) ? 1 - u : u;
      v = u32IsVInverted(texture_index) ? 1 - v : v;

      if constexpr (core::build::is_gpu_build) {
        if (use_interop) {
#ifdef __CUDA_ARCH__
          {
            device::gpgpu::APITextureHandle sampler2D = bundle.getDeviceTextureHandle(texture_index, TextureBundleViews::UINT32);
            uchar4 pixel = tex2D<uchar4>(sampler2D, u, v);
            return (pixel.w << 24) | (pixel.x << 16) | (pixel.y << 8) | pixel.z;
          }
#endif
          //  Other backend texture sampling methods need to be implemented here.
        }
      }
      const int width = u32width(texture_index);
      const int height = u32height(texture_index);
      unsigned i = uv2index(u, width - 1);
      unsigned j = uv2index(v, height - 1);
      unsigned idx = (j * width + i);
      AX_ASSERT_LT(idx, width * height);
      return u32texture(texture_index).raw_data[idx];
    }

    /* Outputs pixel in BGRA format. */
    ax_device_callable_inlined glm::vec4 f32pixel(uint64_t texture_index, float u, float v) const {
      u = f32IsUInverted(texture_index) ? 1 - u : u;
      v = f32IsVInverted(texture_index) ? 1 - v : v;

      if constexpr (core::build::is_gpu_build) {
        if (use_interop) {
#ifdef __CUDA_ARCH__
          device::gpgpu::APITextureHandle sampler2D = bundle.getDeviceTextureHandle(texture_index, TextureBundleViews::FLOAT32);
          float4 pixel = tex2D<float4>(sampler2D, u, v);
          return {pixel.x, pixel.y, pixel.z, pixel.w};
#endif
          //  Other backend texture sampling methods need to be implemented here.
        }
      }
      const int width = f32width(texture_index);
      const int height = f32height(texture_index);
      unsigned i = uv2index(u, width - 1);
      unsigned j = uv2index(v, height - 1);
      const int channels = f32channels(texture_index);
      unsigned idx = (j * width + i) * channels;
      AX_ASSERT_LT(idx + channels - 1, width * height * channels);
      glm::vec4 rgba;
      F32Texture sampled_texture = f32texture(texture_index);
      for (int p = 0; p < channels; p++)
        rgba[p] = sampled_texture.raw_data[idx + p];

      return rgba;
    }
  };
}  // namespace nova::texturing
#endif  // TEXTURECONTEXT_H
