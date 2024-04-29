#ifndef NOVA_TEXTURING_H
#define NOVA_TEXTURING_H
#include "Axomae_macros.h"
#include "TextureProcessing.h"
#include "math_utils.h"

namespace nova::texturing {
  struct EnvmapResourcesHolder {
    TextureOperations<float> texture_processor;
  };

  inline glm::vec3 sample_cubemap(const glm::vec3 &sample_vector, const EnvmapResourcesHolder *res_holder) {
    AX_ASSERT(res_holder != nullptr, "");
    glm::vec3 normalized = glm::normalize(sample_vector);
    const glm::vec2 sph = math::spherical::cartesianToSpherical(normalized);
    const glm::vec2 uv = math::spherical::sphericalToUv(sph);

    return res_holder->texture_processor.uvSample(uv.x, 1 - uv.y);
  }

  inline glm::vec3 sample_cubemap_plane(const glm::vec3 &sample_vector, const glm::vec3 &up_vector, const EnvmapResourcesHolder *res_holder) {
    const glm::vec3 same_plane_vector = glm::dot(sample_vector, up_vector) * sample_vector;
    return sample_cubemap(same_plane_vector, res_holder);
  }

}  // namespace nova::texturing
#endif  // NOVA_TEXTURING_H
