#ifndef TEXTURECONTEXT_H
#define TEXTURECONTEXT_H
#include "texture_datastructures.h"

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
    ax_device_callable TextureBundleViews(const u32tex_shared_views_s &u32_textures);
    ax_device_callable TextureBundleViews(const u32tex_shared_views_s &ftex, const f32tex_shared_views_s &utex);
    ax_device_callable const U32Texture &getU32(std::size_t index) const;
    ax_device_callable const F32Texture &getF32(std::size_t index) const;
    ax_device_callable const device::gpgpu::APITextureHandle &getDeviceTextureHandle(std::size_t index, FORMAT format) const;
    ax_device_callable int getChannelsU32(std::size_t index) const;
    ax_device_callable int getChannelsF32(std::size_t index) const;
  };

  class TextureCtx {
    TextureBundleViews bundle;
    bool use_interop{true};

   public:
    CLASS_DCM(TextureCtx)

    ax_device_callable TextureCtx(const TextureBundleViews &bundle, bool use_interop = true);
    ax_device_callable const U32Texture &u32texture(std::size_t index) const;
    ax_device_callable const F32Texture &f32texture(std::size_t index) const;
    ax_device_callable int u32width(std::size_t index) const;
    ax_device_callable int u32height(std::size_t index) const;
    ax_device_callable uint32_t u32pixel(std::size_t texture_index, float u, float v) const;
    ax_device_callable int u32channels(std::size_t texture_index) const;
    ax_device_callable const TextureBundleViews &getBundle() const { return bundle; }
  };
}  // namespace nova::texturing
#endif  // TEXTURECONTEXT_H
