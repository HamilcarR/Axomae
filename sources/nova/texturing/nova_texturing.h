#ifndef NOVA_TEXTURING_H
#define NOVA_TEXTURING_H
#include "Axomae_macros.h"
#include "NovaTextures.h"
#include "math_utils.h"
#include "utils/macros.h"
#include <memory>

namespace nova::texturing {

  struct TextureResourcesHolder {
    std::vector<std::unique_ptr<NovaTextureInterface>> textures;

    REGISTER_RESOURCE(texture, NovaTextureInterface, textures)
  };
  RESOURCES_DEFINE_CREATE(NovaTextureInterface)

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
