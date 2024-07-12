#ifndef NOVA_TEXTURING_H
#define NOVA_TEXTURING_H
#include "ConstantTexture.h"
#include "ImageTexture.h"
#include "math_utils.h"
#include "project_macros.h"
#include "utils/macros.h"
#include <memory>

namespace nova::texturing {

  struct TextureResourcesHolder {
    std::vector<std::unique_ptr<NovaTextureInterface>> textures;

    REGISTER_RESOURCE(texture, NovaTextureInterface, textures)
  };
  RESOURCES_DEFINE_CREATE(NovaTextureInterface)

  glm::vec3 sample_cubemap(const glm::vec3 &sample_vector, const TextureRawData *res_holder);
  glm::vec3 sample_cubemap_plane(const glm::vec3 &sample_vector, const glm::vec3 &up_vector, const TextureRawData *res_holder);

}  // namespace nova::texturing
#endif  // NOVA_TEXTURING_H
