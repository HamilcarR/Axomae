#include "nova_texturing.h"
namespace nova::texturing {

  /* returns uv in [0 , 1] interval*/
  float normalize_uv(float uv) {
    if (uv < 0) {
      float float_part = std::ceil(uv) - uv;
      return 1.f - float_part;
    }
    return uv;
  }

  glm::vec3 sample_cubemap(const glm::vec3 &sample_vector, const TextureRawData *res_holder) {
    AX_ASSERT_NOTNULL(res_holder);
    AX_ASSERT_NOTNULL(res_holder->raw_data);
    glm::vec3 normalized = glm::normalize(sample_vector);
    std::swap(normalized.y, normalized.z);
    normalized.z = -normalized.z;
    const glm::vec2 sph = math::spherical::cartesianToSpherical(normalized);
    const glm::vec2 uv = math::spherical::sphericalToUv(sph);
    const float u = normalize_uv(uv.x);
    const float v = normalize_uv(uv.y);

    const int x = (int)math::texture::uvToPixel(u, res_holder->width - 1);
    const int y = (int)math::texture::uvToPixel(1 - v, res_holder->height - 1);

    int index = (y * res_holder->width + x) * res_holder->channels;
    AX_ASSERT_LT(index + 2, res_holder->width * res_holder->height * res_holder->channels);
    const std::vector<float> &data = *res_holder->raw_data;
    return {data[index], data[index + 1], data[index + 2]};  // TODO : sync threads before swap ?
  }

  glm::vec3 sample_cubemap_plane(const glm::vec3 &sample_vector, const glm::vec3 &up_vector, const TextureRawData *res_holder) {
    const glm::vec3 same_plane_vector = glm::dot(sample_vector, up_vector) * sample_vector;
    return sample_cubemap(same_plane_vector, res_holder);
  }
}  // namespace nova::texturing