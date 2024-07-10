#include "nova_texturing.h"
namespace nova::texturing {
  glm::vec3 sample_cubemap(const glm::vec3 &sample_vector, const TextureRawData *res_holder) {
    AX_ASSERT(res_holder != nullptr, "");
    glm::vec3 normalized = glm::normalize(sample_vector);
    std::swap(normalized.y, normalized.z);
    normalized.z = -normalized.z;
    const glm::vec2 sph = math::spherical::cartesianToSpherical(normalized);
    const glm::vec2 uv = math::spherical::sphericalToUv(sph);
    const float u = uv.x;
    const float v = uv.y;

    const int x = math::texture::uvToPixel(u, res_holder->width - 1);
    const int y = math::texture::uvToPixel(1 - v, res_holder->height - 1);

    int index = (y * res_holder->width + x) * res_holder->channels;
    const float *data = res_holder->raw_data->data();
    return {data[index], data[index + 1], data[index + 2]};
  }

  glm::vec3 sample_cubemap_plane(const glm::vec3 &sample_vector, const glm::vec3 &up_vector, const TextureRawData *res_holder) {
    const glm::vec3 same_plane_vector = glm::dot(sample_vector, up_vector) * sample_vector;
    return sample_cubemap(same_plane_vector, res_holder);
  }
}  // namespace nova::texturing