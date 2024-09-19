#ifndef UTILS_3D_H
#define UTILS_3D_H
#include "math_utils.h"
#include "vector/Vector.h"

namespace math::geometry {
  /* Get barycentric coordinates of I in triangle P1P2P3 */
  ax_device_callable_inlined Vec3f barycentric_lerp(const Vec2f &P1, const Vec2f &P2, const Vec2f &P3, const Vec2f &I) {
    float W1 = ((P2.y - P3.y) * (I.x - P3.x) + (P3.x - P2.x) * (I.y - P3.y)) / ((P2.y - P3.y) * (P1.x - P3.x) + (P3.x - P2.x) * (P1.y - P3.y));
    float W2 = ((P3.y - P1.y) * (I.x - P3.x) + (P1.x - P3.x) * (I.y - P3.y)) / ((P2.y - P3.y) * (P1.x - P3.x) + (P3.x - P2.x) * (P1.y - P3.y));
    float W3 = 1 - W1 - W2;
    return {W1, W2, W3};
  }

  template<class TYPE>
  ax_device_callable_inlined TYPE barycentric_lerp(const TYPE &p0, const TYPE &p1, const TYPE &p2, float w, float u, float v) {
    return p0 * w + p1 * u + p2 * v;
  }
  /* Builds a mat4 in column major */
  ax_device_callable_inlined glm::mat4 construct_transformation_matrix(const glm::vec3 &xaxis,
                                                                       const glm::vec3 &yaxis,
                                                                       const glm::vec3 &zaxis,
                                                                       const glm::vec3 &transl) {

    return glm::mat4({xaxis, 0.f}, {yaxis, 0.f}, {zaxis, 0.f}, {transl, 1.f});
  }
  ax_device_callable_inlined glm::mat3 compute_normal_mat(const glm::mat4 &model) { return glm::transpose(glm::inverse(glm::mat3(model))); }
  ax_device_callable_inlined glm::mat3 glm_extract_rotation(const glm::mat4 &model) { return {model}; }
  ax_device_callable_inlined glm::vec3 glm_extract_translation(const glm::mat4 &model) { return {model[3]}; }
  ax_device_callable_inlined glm::vec3 glm_extract_scale(const glm::mat4 &model) { return {model[0].x, model[1].y, model[2].z}; }
  ax_device_callable_inlined glm::vec3 glm_extract_xaxis(const glm::mat4 &model) { return {glm::normalize(model[0])}; }
  ax_device_callable_inlined glm::vec3 glm_extract_yaxis(const glm::mat4 &model) { return {glm::normalize(model[1])}; }
  ax_device_callable_inlined glm::vec3 glm_extract_zaxis(const glm::mat4 &model) { return {glm::normalize(model[2])}; }
  ax_device_callable_inlined glm::mat3 construct_tbn(const glm::vec3 &normal, const glm::vec3 &tangent, const glm::vec3 &bitangent) {
    return {tangent, bitangent, normal};
  }
  ax_device_callable_inlined glm::mat3 construct_tbn(const glm::vec3 &normal) {
    glm::vec3 up{0.f, 1.f, 0.f};
    glm::vec3 tangent = glm::cross(normal, up);
    glm::vec3 bitangent = glm::cross(normal, tangent);
    return {tangent, bitangent, normal};
  }

}  // namespace math::geometry

#endif
