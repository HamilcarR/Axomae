#ifndef NOVA_TEXTURING_H
#define NOVA_TEXTURING_H
#include "Includes.cuh"
#include "OfflineCubemapProcessing.h"
#include "math_utils.h"

namespace nova::texturing {
  struct SceneResourcesHolder {
    EnvmapProcessing<float> texture_processor;
  };

  inline glm::vec3 sample_cubemap(const glm::vec3 &sample_vector, const SceneResourcesHolder *res_holder) {
    glm::vec3 normalized = glm::normalize(sample_vector);
    const glm::vec2 sph = math::spherical::cartesianToSpherical(normalized);
    const glm::vec2 uv = math::spherical::sphericalToUv(sph);
    return res_holder->texture_processor.uvSample(uv.x, uv.y);
  }

  inline glm::vec3 compute_envmap_background(const Ray &r, const SceneResourcesHolder *res_holder) {
    AX_ASSERT(res_holder != nullptr, "");
    return sample_cubemap(r.direction, res_holder);
  }

}  // namespace nova::texturing
#endif  // NOVA_TEXTURING_H
