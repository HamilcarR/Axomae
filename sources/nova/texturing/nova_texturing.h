#ifndef NOVA_TEXTURING_H
#define NOVA_TEXTURING_H
#include "Axomae_macros.h"
#include "math_utils.h"

namespace nova::texturing {
  struct TextureResourcesHolder {
    std::vector<float> *raw_data;
    int width;
    int height;
    int channels;
  };

  glm::vec3 sample_cubemap(const glm::vec3 &sample_vector, const TextureResourcesHolder *res_holder);

  glm::vec3 sample_cubemap_plane(const glm::vec3 &sample_vector, const glm::vec3 &up_vector, const TextureResourcesHolder *res_holder);

}  // namespace nova::texturing
#endif  // NOVA_TEXTURING_H
