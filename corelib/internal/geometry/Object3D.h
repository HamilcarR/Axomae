#ifndef OBJECT3D_H
#define OBJECT3D_H
#include "internal/device/gpgpu/device_utils.h"
#include <internal/common/axstd/span.h>
#include <internal/common/math/math_utils.h>
#include <internal/macro/project_macros.h>

/* Triangle based mesh parameters. Would probably need a renaming.*/

struct vertices_attrb3d_t {
  glm::vec3 v0;
  glm::vec3 v1;
  glm::vec3 v2;
};

struct vertices_attrb2d_t {
  glm::vec2 v0;
  glm::vec2 v1;
  glm::vec2 v2;
};

struct edges_t {
  glm::vec3 e1;
  glm::vec3 e2;
};

namespace geometry {

  struct face_data_tri {
    float v0[3], v1[3], v2[3];        // vertices
    float n0[3], n1[3], n2[3];        // normals
    float c0[3], c1[3], c2[3];        // colors
    float uv0[2], uv1[2], uv2[2];     // texture coordinates
    float tan0[3], tan1[3], tan2[3];  // tangent
    float bit0[3], bit1[3], bit2[3];  // bitangents

    /* n_transform is transpose(inverse(transform)) */
    ax_device_callable_inlined void transform(const glm::mat4 &transform, const glm::mat3 &n_transform) {
      auto transformVec = [&](float v[3]) {
        glm::vec3 vec{v[0], v[1], v[2]};
        vec = transform * glm::vec4(vec, 1.f);
        v[0] = vec.x;
        v[1] = vec.y;
        v[2] = vec.z;
      };

      // Lambda for transforming directions (normals, tangents, bitangents) using the normal matrix.
      auto transformDir = [&](float d[3]) {
        glm::vec3 vec{d[0], d[1], d[2]};
        vec = n_transform * vec;
        d[0] = vec.x;
        d[1] = vec.y;
        d[2] = vec.z;
      };

      // Transform vertices.
      transformVec(v0);
      transformVec(v1);
      transformVec(v2);

      // Transform normals.
      transformDir(n0);
      transformDir(n1);
      transformDir(n2);

      // Transform tangents.
      transformDir(tan0);
      transformDir(tan1);
      transformDir(tan2);

      // Transform bitangents.
      transformDir(bit0);
      transformDir(bit1);
      transformDir(bit2);
    }

    ax_device_callable_inlined vertices_attrb3d_t vertices() const {
      return vertices_attrb3d_t{glm::vec3(v0[0], v0[1], v0[2]), glm::vec3(v1[0], v1[1], v1[2]), glm::vec3(v2[0], v2[1], v2[2])};
    }

    ax_device_callable_inlined vertices_attrb2d_t uvs() const {
      return vertices_attrb2d_t{glm::vec2(uv0[0], uv0[1]), glm::vec2(uv1[0], uv1[1]), glm::vec2(uv2[0], uv2[1])};
    }

    ax_device_callable_inlined vertices_attrb3d_t normals() const {
      return vertices_attrb3d_t{glm::vec3(n0[0], n0[1], n0[2]), glm::vec3(n1[0], n1[1], n1[2]), glm::vec3(n2[0], n2[1], n2[2])};
    }

    ax_device_callable_inlined vertices_attrb3d_t colors() const {
      return vertices_attrb3d_t{glm::vec3(c0[0], c0[1], c0[2]), glm::vec3(c1[0], c1[1], c1[2]), glm::vec3(c2[0], c2[1], c2[2])};
    }

    ax_device_callable_inlined vertices_attrb3d_t tangents() const {
      return vertices_attrb3d_t{glm::vec3(tan0[0], tan0[1], tan0[2]), glm::vec3(tan1[0], tan1[1], tan1[2]), glm::vec3(tan2[0], tan2[1], tan2[2])};
    }

    ax_device_callable_inlined vertices_attrb3d_t bitangents() const {
      return vertices_attrb3d_t{glm::vec3(bit0[0], bit0[1], bit0[2]), glm::vec3(bit1[0], bit1[1], bit1[2]), glm::vec3(bit2[0], bit2[1], bit2[2])};
    }

    ax_device_callable_inlined vertices_attrb3d_t vertices(const glm::mat4 &transform) const {
      return vertices_attrb3d_t{transform * glm::vec4(v0[0], v0[1], v0[2], 1.f),
                                transform * glm::vec4(v1[0], v1[1], v1[2], 1.f),
                                transform * glm::vec4(v2[0], v2[1], v2[2], 1.f)};
    }

    ax_device_callable_inlined vertices_attrb3d_t normals(const glm::mat3 &transform) const {
      return vertices_attrb3d_t{
          transform * glm::vec3(n0[0], n0[1], n0[2]), transform * glm::vec3(n1[0], n1[1], n1[2]), transform * glm::vec3(n2[0], n2[1], n2[2])};
    }

    ax_device_callable_inlined vertices_attrb3d_t tangents(const glm::mat3 &transform) const {
      return vertices_attrb3d_t{transform * glm::vec3(tan0[0], tan0[1], tan0[2]),
                                transform * glm::vec3(tan1[0], tan1[1], tan1[2]),
                                transform * glm::vec3(tan2[0], tan2[1], tan2[2])};
    }

    ax_device_callable_inlined vertices_attrb3d_t bitangents(const glm::mat3 &transform) const {
      return vertices_attrb3d_t{transform * glm::vec3(bit0[0], bit0[1], bit0[2]),
                                transform * glm::vec3(bit1[0], bit1[1], bit1[2]),
                                transform * glm::vec3(bit2[0], bit2[1], bit2[2])};
    }

    ax_device_callable_inlined glm::vec3 compute_center() {
      float x = (v0[0] + v1[0] + v2[0]) * 0.333333f;
      float y = (v0[1] + v1[1] + v2[1]) * 0.333333f;
      float z = (v0[2] + v1[2] + v2[2]) * 0.333333f;
      return {x, y, z};
    }

    ax_device_callable_inlined edges_t compute_edges() {
      edges_t edges{};
      edges.e1 = glm::vec3(v1[0], v1[1], v1[2]) - glm::vec3(v0[0], v0[1], v0[2]);
      edges.e2 = glm::vec3(v2[0], v2[1], v2[2]) - glm::vec3(v0[0], v0[1], v0[2]);
      return edges;
    }

    ax_device_callable_inlined vertices_attrb3d_t compute_bitangents() {
      vertices_attrb3d_t bitans{};
      glm::vec3 n_0 = {n0[0], n0[1], n0[2]};
      glm::vec3 n_1 = {n1[0], n1[1], n1[2]};
      glm::vec3 n_2 = {n2[0], n2[1], n2[2]};
      glm::vec3 tan_0 = {tan0[0], tan0[1], tan0[2]};
      glm::vec3 tan_1 = {tan1[0], tan1[1], tan1[2]};
      glm::vec3 tan_2 = {tan2[0], tan2[1], tan2[2]};

      bitans.v0 = glm::normalize(glm::cross(n_0, tan_0));
      bitans.v1 = glm::normalize(glm::cross(n_1, tan_1));
      bitans.v2 = glm::normalize(glm::cross(n_2, tan_2));

      return bitans;
    }

    ax_device_callable_inlined vertices_attrb3d_t compute_tangents() {
      vertices_attrb3d_t tans{};
      glm::vec3 n_0 = {n0[0], n0[1], n0[2]};
      glm::vec3 n_1 = {n1[0], n1[1], n1[2]};
      glm::vec3 n_2 = {n2[0], n2[1], n2[2]};
      glm::vec3 bitan0 = {bit0[0], bit0[1], bit0[2]};
      glm::vec3 bitan1 = {bit1[0], bit1[1], bit1[2]};
      glm::vec3 bitan2 = {bit2[0], bit2[1], bit2[2]};

      tans.v0 = glm::normalize(glm::cross(n_0, bitan0));
      tans.v1 = glm::normalize(glm::cross(n_1, bitan1));
      tans.v2 = glm::normalize(glm::cross(n_2, bitan2));

      return tans;
    }

    ax_device_callable_inlined bool hasValidUvs() const {
      bool tex0 = uv0[0] != 0 || uv0[1] != 0;
      bool tex1 = uv1[0] != 0 || uv1[1] != 0;
      bool tex2 = uv2[0] != 0 || uv2[1] != 0;
      return tex0 || tex1 || tex2;
    }

    ax_device_callable_inlined bool hasValidNormals() const {
      bool nml0 = n0[0] != 0 || n0[1] != 0 || n0[2] != 0;
      bool nml1 = n1[0] != 0 || n1[1] != 0 || n1[2] != 0;
      bool nml2 = n2[0] != 0 || n2[1] != 0 || n2[2] != 0;
      return nml0 || nml1 || nml2;
    }

    ax_device_callable_inlined bool hasValidTangents() const {
      bool t0 = tan0[0] != 0 || tan0[1] != 0 || tan0[2] != 0;
      bool t1 = tan1[0] != 0 || tan1[1] != 0 || tan1[2] != 0;
      bool t2 = tan2[0] != 0 || tan2[1] != 0 || tan2[2] != 0;
      return t0 || t1 || t2;
    }

    ax_device_callable_inlined bool hasValidBitangents() const {
      bool t0 = bit0[0] != 0 || bit0[1] != 0 || bit0[2] != 0;
      bool t1 = bit1[0] != 0 || bit1[1] != 0 || bit1[2] != 0;
      bool t2 = bit2[0] != 0 || bit2[1] != 0 || bit2[2] != 0;
      return t0 || t1 || t2;
    }
  };
  template<class T>
  ax_device_callable void load_vertex_attribute3f(T c0[3], T c1[3], T c2[3], const unsigned idx[3], const axstd::span<T> &attribute) {
    if (attribute.empty())
      return;
    for (int i = 0; i < 3; i++) {
      c0[i] = attribute[idx[0] * 3 + i];
      c1[i] = attribute[idx[1] * 3 + i];
      c2[i] = attribute[idx[2] * 3 + i];
    }
  }

  template<class T>
  ax_device_callable void load_vertex_attribute2f(T c0[2], T c1[2], T c2[2], const unsigned idx[3], const axstd::span<T> &attribute) {
    if (attribute.empty())
      return;
    for (int i = 0; i < 2; i++) {
      c0[i] = attribute[idx[0] * 2 + i];
      c1[i] = attribute[idx[1] * 2 + i];
      c2[i] = attribute[idx[2] * 2 + i];
    }
  }

  ax_device_callable_inlined void transform_vertices(const face_data_tri &tri_primitive, const glm::mat4 &final_transfo, glm::vec3 vertices[3]) {
    glm::vec3 v1{tri_primitive.v0[0], tri_primitive.v0[1], tri_primitive.v0[2]};
    glm::vec3 v2{tri_primitive.v1[0], tri_primitive.v1[1], tri_primitive.v1[2]};
    glm::vec3 v3{tri_primitive.v2[0], tri_primitive.v2[1], tri_primitive.v2[2]};

    vertices[0] = final_transfo * glm::vec4(v1, 1.f);
    vertices[1] = final_transfo * glm::vec4(v2, 1.f);
    vertices[2] = final_transfo * glm::vec4(v3, 1.f);
  }

  ax_device_callable_inlined void transform_normals(const face_data_tri &tri_primitive, const glm::mat3 &normal_matrix, glm::vec3 normals[3]) {

    glm::vec3 n1{tri_primitive.n0[0], tri_primitive.n0[1], tri_primitive.n0[2]};
    glm::vec3 n2{tri_primitive.n1[0], tri_primitive.n1[1], tri_primitive.n1[2]};
    glm::vec3 n3{tri_primitive.n2[0], tri_primitive.n2[1], tri_primitive.n2[2]};

    normals[0] = normal_matrix * n1;
    normals[1] = normal_matrix * n2;
    normals[2] = normal_matrix * n3;
  }

  ax_device_callable_inlined void transform_tangents(const face_data_tri &tri_primitive, const glm::mat3 &normal_matrix, glm::vec3 tangents[3]) {
    glm::vec3 t0{tri_primitive.tan0[0], tri_primitive.tan0[1], tri_primitive.tan0[2]};
    glm::vec3 t1{tri_primitive.tan1[0], tri_primitive.tan1[1], tri_primitive.tan1[2]};
    glm::vec3 t2{tri_primitive.tan2[0], tri_primitive.tan2[1], tri_primitive.tan2[2]};

    tangents[0] = normal_matrix * t0;
    tangents[1] = normal_matrix * t1;
    tangents[2] = normal_matrix * t2;
  }
  ax_device_callable_inlined void transform_bitangents(const face_data_tri &tri_primitive, const glm::mat3 &normal_matrix, glm::vec3 bitangents[3]) {
    glm::vec3 b0{tri_primitive.bit0[0], tri_primitive.bit0[1], tri_primitive.bit0[2]};
    glm::vec3 b1{tri_primitive.bit1[0], tri_primitive.bit1[1], tri_primitive.bit1[2]};
    glm::vec3 b2{tri_primitive.bit2[0], tri_primitive.bit2[1], tri_primitive.bit2[2]};

    bitangents[0] = normal_matrix * b0;
    bitangents[1] = normal_matrix * b1;
    bitangents[2] = normal_matrix * b2;
  }

  ax_device_callable_inlined void normalize_uv(glm::vec2 &textures) {
    if (textures.s > 1 || textures.s < 0)
      textures.s = textures.s - std::floor(textures.s);
    if (textures.t > 1 || textures.t < 0)
      textures.t = textures.t - std::floor(textures.t);
  }

  ax_device_callable_inlined void extract_uvs(const face_data_tri &tri_primitive, glm::vec2 textures[3]) {
    textures[0] = {tri_primitive.uv0[0], tri_primitive.uv0[1]};
    textures[1] = {tri_primitive.uv1[0], tri_primitive.uv1[1]};
    textures[2] = {tri_primitive.uv2[0], tri_primitive.uv2[1]};
    normalize_uv(textures[0]);
    normalize_uv(textures[1]);
    normalize_uv(textures[2]);
  }

  ax_device_callable_inlined void transform_attr(vertices_attrb3d_t &attr, const glm::mat4 &matrix) {
    attr.v0 = matrix * glm::vec4(attr.v0, 1.f);
    attr.v1 = matrix * glm::vec4(attr.v1, 1.f);
    attr.v2 = matrix * glm::vec4(attr.v2, 1.f);
  }

  ax_device_callable_inlined void transform_attr(vertices_attrb3d_t &attr, const glm::mat3 &matrix) {
    attr.v0 = matrix * attr.v0;
    attr.v1 = matrix * attr.v1;
    attr.v2 = matrix * attr.v2;
  }

  ax_device_callable_inlined edges_t compute_edges(const vertices_attrb3d_t &vertices) {
    edges_t edges{};
    edges.e1 = vertices.v1 - vertices.v0;
    edges.e2 = vertices.v2 - vertices.v0;
    return edges;
  }
}  // namespace geometry

// TODO: Triangle mesh , replace name
class Object3D {
 public:
  axstd::span<float> vertices{};
  axstd::span<float> uv{};
  axstd::span<float> colors{};
  axstd::span<float> normals{};
  axstd::span<float> bitangents{};
  axstd::span<float> tangents{};
  axstd::span<unsigned> indices{};

  constexpr static int face_stride = 3;

  ax_device_callable void getTri(geometry::face_data_tri &geom, const unsigned int idx[3]) const {

    for (int i = 0; i < 3; i++) {
      if (!vertices.empty()) {
        geom.v0[i] = vertices[idx[0] * face_stride + i];
        geom.v1[i] = vertices[idx[1] * face_stride + i];
        geom.v2[i] = vertices[idx[2] * face_stride + i];
      }
      if (!normals.empty()) {
        geom.n0[i] = normals[idx[0] * face_stride + i];
        geom.n1[i] = normals[idx[1] * face_stride + i];
        geom.n2[i] = normals[idx[2] * face_stride + i];
      }
      if (!colors.empty()) {
        geom.c0[i] = colors[idx[0] * face_stride + i];
        geom.c1[i] = colors[idx[1] * face_stride + i];
        geom.c2[i] = colors[idx[2] * face_stride + i];
      }
      if (!tangents.empty()) {
        geom.tan0[i] = tangents[idx[0] * face_stride + i];
        geom.tan1[i] = tangents[idx[1] * face_stride + i];
        geom.tan2[i] = tangents[idx[2] * face_stride + i];
      }
      if (!bitangents.empty()) {
        geom.bit0[i] = bitangents[idx[0] * face_stride + i];
        geom.bit1[i] = bitangents[idx[1] * face_stride + i];
        geom.bit2[i] = bitangents[idx[2] * face_stride + i];
      }
    }
    if (!uv.empty()) {
      for (int i = 0; i < 2; i++) {
        geom.uv0[i] = uv[idx[0] * 2 + i];
        geom.uv1[i] = uv[idx[1] * 2 + i];
        geom.uv2[i] = uv[idx[2] * 2 + i];
      }
    }
  }

  ax_device_callable geometry::face_data_tri getFace(std::size_t triangle_id) const {
    geometry::face_data_tri tri_primitive{};
    AX_ASSERT_FALSE(indices.empty());
    unsigned idx[3] = {indices[triangle_id], indices[triangle_id + 1], indices[triangle_id + 2]};
    getTri(tri_primitive, idx);
    return tri_primitive;
  }
};

#endif
