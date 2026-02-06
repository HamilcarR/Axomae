#ifndef UTILS_3D_H
#define UTILS_3D_H
#include "math_utils.h"
#include "vector/Vector.h"

namespace math {
  namespace geometry {
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
  }  // namespace geometry

  template<class T>
  constexpr T halfvector(const T &v1, const T &v2) {
    return glm::normalize(v1 + v2);
  }

  /* Recomputes orthonormal base to ensure perfect orthonormality (Gram-Shmidt).*/
  ax_device_callable_inlined void gram_shmidt(const glm::vec3 &n, const glm::vec3 &t, glm::vec3 &n_res, glm::vec3 &t_res, glm::vec3 &b_res) {
    n_res = glm::normalize(n);
    glm::vec3 t_ortho = glm::normalize(t - glm::dot(t, n_res) * n_res);
    if (glm::length2(t_ortho) < 1e-10f) {
      t_ortho = glm::abs(n_res.x) > 0.9f ? glm::vec3(0, 1, 0) : glm::vec3(1, 0, 0);
      t_ortho -= glm::dot(t_ortho, n_res) * n_res;
    }
    t_res = glm::normalize(t_ortho);
    b_res = glm::cross(n_res, t_res);
  }

  ax_device_callable_inlined glm::mat3 gram_shmidt(const glm::vec3 &normal, const glm::vec3 &tangent, const glm::vec3 &bitangent) {
    glm::vec3 N = glm::normalize(normal);
    glm::vec3 T = glm::normalize(tangent - glm::dot(tangent, N) * N);
    glm::vec3 B = glm::cross(N, T);
    if (glm::dot(B, bitangent) < 0.0f)
      B = -B;

    return glm::mat3(T, B, N);
  }

  /* Creates one valid orthonormal coordinate system. (see Building An Orthonormal Basis, Revisited.)*/
  ax_device_callable_inlined void make_onb(glm::vec3 n, glm::vec3 &b1, glm::vec3 &b2) {
    n = glm::normalize(n);
    float sign = copysignf(1.f, n.z);
    float a = -1.f / (sign + n.z);
    float b = n.x * n.y * a;
    b1 = {1.f + sign * n.x * n.x * a, sign * b, -sign * n.x};
    b2 = {b, sign + n.y * n.y * a, -n.y};
  }
}  // namespace math

struct transform4x4_t {
  glm::mat4 m;
  glm::mat4 inv;  // inverse
  glm::mat4 t;    // transpose
  glm::mat3 n;    // normal ( mat3(transpose(invert(m)) )
  ax_device_callable bool operator==(const transform4x4_t &other) const {
    // no need to compare the others , waste of cycles. If the other matrices are not equal, we raise this in the assert.
    bool equal = m == other.m;
    AX_ASSERT(!equal || (inv == other.inv && t == other.t && n == other.n), "Invalid transform matrix");
    return equal;
  }
  ax_device_callable static constexpr std::size_t padding() { return 57; }  // how many elements in the record
};

struct transform3x3_t {
  glm::mat3 m;
  glm::mat3 inv;
  glm::mat3 t;
  bool operator==(const transform3x3_t &other) const { return m == other.m; }
  static constexpr std::size_t padding() { return 27; }
};

#endif
