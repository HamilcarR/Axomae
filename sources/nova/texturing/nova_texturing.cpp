#include "nova_texturing.h"
#include "TextureProcessing.h"
namespace nova::texturing {
  glm::vec3 sample_cubemap(const glm::vec3 &sample_vector, const TextureRawData *res_holder) {
    AX_ASSERT(res_holder != nullptr, "");
    const TextureOperations<float> texture_processor(*res_holder->raw_data, res_holder->width, res_holder->height);
    glm::vec3 normalized = glm::normalize(sample_vector);
    std::swap(normalized.y, normalized.z);
    normalized.z = -normalized.z;
    const glm::vec2 sph = math::spherical::cartesianToSpherical(normalized);
    const glm::vec2 uv = math::spherical::sphericalToUv(sph);

    return texture_processor.uvSample(uv.x, 1 - uv.y);
  }

  glm::vec3 sample_cubemap_plane(const glm::vec3 &sample_vector, const glm::vec3 &up_vector, const TextureRawData *res_holder) {
    const glm::vec3 same_plane_vector = glm::dot(sample_vector, up_vector) * sample_vector;
    return sample_cubemap(same_plane_vector, res_holder);
  }
}  // namespace nova::texturing