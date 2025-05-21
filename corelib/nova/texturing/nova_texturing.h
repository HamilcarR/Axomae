#ifndef NOVA_TEXTURING_H
#define NOVA_TEXTURING_H
#include "NovaTextureInterface.h"
#include "TextureContext.h"
#include "texture_interop_storage.h"
#include <internal/common/axstd/managed_buffer.h>
#include <internal/common/math/math_utils.h>
#include <internal/common/type_list.h>
#include <internal/common/utils.h>
#include <internal/macro/project_macros.h>

namespace nova::texturing {

  struct texture_init_record_s {
    std::size_t total_constant_textures;
    std::size_t total_image_textures;
  };

  class TextureStorage {
    axstd::managed_vector<NovaTextureInterface> textures;
    axstd::managed_vector<ConstantTexture> constant_textures;
    axstd::managed_vector<ImageTexture> image_textures;
    axstd::managed_vector<EnvmapTexture> envmap_textures;  // Used for environment map blending.

   public:
    CLASS_CM(TextureStorage)

    NovaTextureInterface add(const ConstantTexture &constant_tex) { return append(constant_tex, constant_textures); }
    void allocConstant(std::size_t total_cste_tex);
    bool isConstantEmpty() const { return constant_textures.empty(); }
    bool isConstantInit() const { return constant_textures.capacity() > 0; }
    std::size_t sizeConstant() const { return constant_textures.size(); }
    CstTexCollection constants() const;
    void clearConstant();

    NovaTextureInterface add(const ImageTexture &image_tex) { return append(image_tex, image_textures); }
    void allocImage(std::size_t total_img_tex);
    bool isImageEmpty() const { return image_textures.empty(); }
    bool isImageInit() const { return image_textures.capacity() > 0; }
    std::size_t sizeImage() const { return image_textures.size(); }
    ImgTexCollection images() const;
    void clearImage();

    NovaTextureInterface add(const EnvmapTexture &envmap_tex) { return append(envmap_tex, envmap_textures); }
    void allocEnvmap(std::size_t total_env_tex);
    bool isEnvmapEmpty() const { return envmap_textures.empty(); }
    bool isEnvmapInit() const { return envmap_textures.capacity() > 0; }
    std::size_t sizeEnvmap() const { return envmap_textures.size(); }
    EnvMapCollection envmaps() const;
    void clearEnvmap();

    IntfTexCollection pointers() const;

    void clear();

   private:
    template<class T>
    NovaTextureInterface append(const T &texture, axstd::managed_vector<T> &texture_vector) {
      AX_ASSERT_GE(texture_vector.capacity(), texture_vector.size() + 1);
      texture_vector.push_back(texture);
      NovaTextureInterface texture_ptr = &texture_vector.back();
      textures.push_back(texture_ptr);
      return texture_ptr;
    }
  };

  class TextureResourcesHolder {
    TextureStorage texture_storage;
    DefaultTextureReferencesStorage texture_raw_data_storage;
    unsigned current_environmentmap_index{0};

   public:
    CLASS_CM(TextureResourcesHolder)

    void allocateMeshTextures(const texture_init_record_s &init_data);
    void allocateEnvironmentMaps(std::size_t num_envmaps);
    void lockResources();
    void releaseResources();
    void mapBuffers();

    template<class T, class... Args>
    NovaTextureInterface addNovaTexture(Args &&...args) {
      static_assert(core::has<T, TYPELIST>::has_type, "Provided type is not a Texture type.");
      T tex = T(std::forward<Args>(args)...);
      return texture_storage.add(tex);
    }

    /* Registers the data of a texture inside one of the u32 or f32 views.
     * tex_id will not be taken into account if not a gpgpu build.
     * Returns the ID of the registered texture in its array.
     */
    template<class T>
    std::size_t addTexture(const T *data, int width, int height, int channels, bool inverted_u = false, bool inverted_v = false, GLuint tex_id = 0) {
      TextureRawData<T> texture_raw_data;
      texture_raw_data.invert_u = inverted_u;
      texture_raw_data.invert_v = inverted_v;
      texture_raw_data.is_rgba = false;  // just letting this here for now, will replace with better memory format system.
      texture_raw_data.width = width;
      texture_raw_data.height = height;
      texture_raw_data.channels = channels;
      texture_raw_data.raw_data = axstd::span<const T>(data, width * height * channels);
      if constexpr (core::build::is_gpu_build)
        return texture_raw_data_storage.add(texture_raw_data, tex_id);
      else
        return texture_raw_data_storage.add(texture_raw_data);
    }

    void clear() {
      texture_storage.clear();
      texture_raw_data_storage.clear();
    }

    TextureBundleViews getTextureBundleViews() const;
    NovaTextureInterface getEnvmap(unsigned index) const;
    NovaTextureInterface getCurrentEnvmap() const;
    void setEnvmapId(unsigned id) { current_environmentmap_index = id; }
  };

}  // namespace nova::texturing
#endif  // NOVA_TEXTURING_H
