#ifndef UTILS_3D_H
#define UTILS_3D_H
#include "Vector.h"
#include "math_utils.h"

namespace math::geometry {
  /* Get barycentric coordinates of I in triangle P1P2P3 */
  inline Vec3f barycentric_lerp(const Vec2f &P1, const Vec2f &P2, const Vec2f &P3, const Vec2f &I) {
    float W1 = ((P2.y - P3.y) * (I.x - P3.x) + (P3.x - P2.x) * (I.y - P3.y)) / ((P2.y - P3.y) * (P1.x - P3.x) + (P3.x - P2.x) * (P1.y - P3.y));
    float W2 = ((P3.y - P1.y) * (I.x - P3.x) + (P1.x - P3.x) * (I.y - P3.y)) / ((P2.y - P3.y) * (P1.x - P3.x) + (P3.x - P2.x) * (P1.y - P3.y));
    float W3 = 1 - W1 - W2;
    return {W1, W2, W3};
  }

  template<class TYPE>
  TYPE barycentric_lerp(const TYPE &p0, const TYPE &p1, const TYPE &p2, float w, float u, float v) {
    return p0 * w + p1 * u + p2 * v;
  }

  inline glm::mat3 compute_normal_mat(const glm::mat4 &model) { return glm::transpose(glm::inverse(glm::mat3(model))); }
  inline glm::mat3 glm_extract_rotation(const glm::mat4 &model) { return {model}; }
  inline glm::vec3 glm_extract_translation(const glm::mat4 &model) { return {model[3]}; }
  inline glm::vec3 glm_extract_scale(const glm::mat4 &model) { return {model[0].x, model[1].y, model[2].z}; }
  inline glm::vec3 glm_extract_xaxis(const glm::mat4 &model) { return {glm::normalize(model[0])}; }
  inline glm::vec3 glm_extract_yaxis(const glm::mat4 &model) { return {glm::normalize(model[1])}; }
  inline glm::vec3 glm_extract_zaxis(const glm::mat4 &model) { return {glm::normalize(model[2])}; }
  inline glm::mat3 construct_tbn(const glm::vec3 &normal, const glm::vec3 &tangent, const glm::vec3 &bitangent) {
    return {tangent, bitangent, normal};
  }
  inline glm::mat3 construct_tbn(const glm::vec3 &normal) {
    glm::vec3 up{0.f, 1.f, 0.f};
    glm::vec3 tangent = glm::cross(normal, up);
    glm::vec3 bitangent = glm::cross(normal, tangent);
    return {tangent, bitangent, normal};
  }

}  // namespace math::geometry

#endif
