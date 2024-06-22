#ifndef NOVA_TEXTURING_H
#define NOVA_TEXTURING_H
#include "Axomae_macros.h"
#include "NovaTextures.h"
#include "math_utils.h"
#include <memory>
namespace nova::texturing {

  struct TextureResourcesHolder {
    std::vector<std::unique_ptr<NovaTextureInterface>> textures;

    template<class TYPE, class... Args>
    NovaTextureInterface *add_texture(Args &&...args) {
      ASSERT_SUBTYPE(NovaTextureInterface, TYPE);
      textures.push_back(std::make_unique<TYPE>(std::forward<Args>(args)...));
      return textures.back().get();
    }
  };

  struct TextureRawData {
    std::vector<float> *raw_data;
    int width;
    int height;
    int channels;
  };

  glm::vec3 sample_cubemap(const glm::vec3 &sample_vector, const TextureRawData *res_holder);
  glm::vec3 sample_cubemap_plane(const glm::vec3 &sample_vector, const glm::vec3 &up_vector, const TextureRawData *res_holder);

}  // namespace nova::texturing
#endif  // NOVA_TEXTURING_H
