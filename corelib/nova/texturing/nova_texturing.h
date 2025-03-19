#ifndef NOVA_TEXTURING_H
#define NOVA_TEXTURING_H
#include "NovaTextureInterface.h"
#include "TextureContext.h"
#include "texture_interop_storage.h"
#include <internal/common/axstd/managed_buffer.h>
#include <internal/common/math/math_utils.h>
#include <internal/common/type_list.h>
#include <internal/macro/project_macros.h>

namespace nova::texturing {

  struct texture_init_record_s {
    std::size_t constant_texture_size;
    std::size_t image_texture_size;
  };

  class TextureStorage {
    axstd::managed_vector<NovaTextureInterface> textures;
    axstd::managed_vector<ConstantTexture> constant_textures;
    axstd::managed_vector<ImageTexture> image_textures;

   public:
    CLASS_CM(TextureStorage)

    NovaTextureInterface add(const ImageTexture &image_tex) { return append(image_tex, image_textures); }
    NovaTextureInterface add(const ConstantTexture &constant_tex) { return append(constant_tex, constant_textures); }

    void allocConstant(std::size_t total_cste_tex) { constant_textures.reserve(total_cste_tex); }
    void allocImage(std::size_t total_cste_tex) { image_textures.reserve(total_cste_tex); }

    void clear() {
      textures.clear();
      constant_textures.clear();
      image_textures.clear();
    }

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
    EnvmapTexture environment_texture;

   public:
    CLASS_CM(TextureResourcesHolder)

    void init(const texture_init_record_s &init_data);
    void lockResources();
    void releaseResources();
    void mapBuffers();

    template<class T, class... Args>
    NovaTextureInterface addNovaTexture(Args &&...args) {
      static_assert(core::has<T, TYPELIST>::has_type, "Provided type is not a Texture type.");
      T tex = T(std::forward<Args>(args)...);
      return texture_storage.add(tex);
    }

    /* tex_id will not be taken into account if not a gpgpu build.*/
    template<class T>
    std::size_t addTexture(const T *data, int width, int height, int channels, GLuint tex_id) {
      TextureRawData<T> texture_raw_data;
      texture_raw_data.width = width;
      texture_raw_data.height = height;
      texture_raw_data.channels = channels;
      texture_raw_data.raw_data = axstd::span<const T>(data, width * height * channels);
#ifdef AXOMAE_USE_CUDA
      return texture_raw_data_storage.add(texture_raw_data, tex_id);
#else
      return texture_raw_data_storage.add(texture_raw_data);
#endif
    }

    void clear() {
      texture_storage.clear();
      texture_raw_data_storage.clear();
      environment_texture = {};
    }

    void setupEnvmap(const float *buffer_image, int width, int height, int channels) {
      environment_texture = EnvmapTexture(buffer_image, width, height, channels);
    }

    TextureBundleViews getTextureBundleViews() const;
    EnvmapTexture getEnvmap() const { return environment_texture; }
  };

}  // namespace nova::texturing
#endif  // NOVA_TEXTURING_H
