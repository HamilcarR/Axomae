#ifndef OBJECT3D_H
#define OBJECT3D_H
#include <internal/common/axstd/span.h>
#include <internal/common/math/math_utils.h>
#include <internal/macro/project_macros.h>

namespace geometry {

  struct face_data_tri {
    float v0[3], v1[3], v2[3];        // vertices
    float n0[3], n1[3], n2[3];        // normals
    float c0[3], c1[3], c2[3];        // colors
    float uv0[2], uv1[2], uv2[2];     // texture coordinates
    float tan0[3], tan1[3], tan2[3];  // tangent
    float bit0[3], bit1[3], bit2[3];  // bitangents
  };
  template<class T>
  ax_device_callable static void load_vertex_attribute3f(T c0[3], T c1[3], T c2[3], const unsigned idx[3], const axstd::span<T> &attribute) {
    if (attribute.empty())
      return;
    for (int i = 0; i < 3; i++) {
      c0[i] = attribute[idx[0] * 3 + i];
      c1[i] = attribute[idx[1] * 3 + i];
      c2[i] = attribute[idx[2] * 3 + i];
    }
  }

  template<class T>
  ax_device_callable static void load_vertex_attribute2f(T c0[2], T c1[2], T c2[2], const unsigned idx[3], const axstd::span<T> &attribute) {
    if (attribute.empty())
      return;
    for (int i = 0; i < 2; i++) {
      c0[i] = attribute[idx[0] * 2 + i];
      c1[i] = attribute[idx[1] * 2 + i];
      c2[i] = attribute[idx[2] * 2 + i];
    }
  }

  ax_device_callable inline void transform_vertices(const face_data_tri &tri_primitive, const glm::mat4 &final_transfo, glm::vec3 vertices[3]) {
    glm::vec3 v1{tri_primitive.v0[0], tri_primitive.v0[1], tri_primitive.v0[2]};
    glm::vec3 v2{tri_primitive.v1[0], tri_primitive.v1[1], tri_primitive.v1[2]};
    glm::vec3 v3{tri_primitive.v2[0], tri_primitive.v2[1], tri_primitive.v2[2]};

    vertices[0] = final_transfo * glm::vec4(v1, 1.f);
    vertices[1] = final_transfo * glm::vec4(v2, 1.f);
    vertices[2] = final_transfo * glm::vec4(v3, 1.f);
  }

  ax_device_callable inline void transform_normals(const face_data_tri &tri_primitive, const glm::mat3 &normal_matrix, glm::vec3 normals[3]) {

    glm::vec3 n1{tri_primitive.n0[0], tri_primitive.n0[1], tri_primitive.n0[2]};
    glm::vec3 n2{tri_primitive.n1[0], tri_primitive.n1[1], tri_primitive.n1[2]};
    glm::vec3 n3{tri_primitive.n2[0], tri_primitive.n2[1], tri_primitive.n2[2]};

    normals[0] = normal_matrix * n1;
    normals[1] = normal_matrix * n2;
    normals[2] = normal_matrix * n3;
  }

  ax_device_callable inline void transform_tangents(const face_data_tri &tri_primitive, const glm::mat3 &normal_matrix, glm::vec3 tangents[3]) {
    glm::vec3 t0{tri_primitive.tan0[0], tri_primitive.tan0[1], tri_primitive.tan0[2]};
    glm::vec3 t1{tri_primitive.tan1[0], tri_primitive.tan1[1], tri_primitive.tan1[2]};
    glm::vec3 t2{tri_primitive.tan2[0], tri_primitive.tan2[1], tri_primitive.tan2[2]};

    tangents[0] = normal_matrix * t0;
    tangents[1] = normal_matrix * t1;
    tangents[2] = normal_matrix * t2;
  }
  ax_device_callable inline void transform_bitangents(const face_data_tri &tri_primitive, const glm::mat3 &normal_matrix, glm::vec3 bitangents[3]) {
    glm::vec3 b0{tri_primitive.bit0[0], tri_primitive.bit0[1], tri_primitive.bit0[2]};
    glm::vec3 b1{tri_primitive.bit1[0], tri_primitive.bit1[1], tri_primitive.bit1[2]};
    glm::vec3 b2{tri_primitive.bit2[0], tri_primitive.bit2[1], tri_primitive.bit2[2]};

    bitangents[0] = normal_matrix * b0;
    bitangents[1] = normal_matrix * b1;
    bitangents[2] = normal_matrix * b2;
  }

  ax_device_callable inline void normalize_uv(glm::vec2 &textures) {
    if (textures.s > 1 || textures.s < 0)
      textures.s = textures.s - std::floor(textures.s);
    if (textures.t > 1 || textures.t < 0)
      textures.t = textures.t - std::floor(textures.t);
  }

  ax_device_callable inline void extract_uvs(const face_data_tri &tri_primitive, glm::vec2 textures[3]) {
    textures[0] = {tri_primitive.uv0[0], tri_primitive.uv0[1]};
    textures[1] = {tri_primitive.uv1[0], tri_primitive.uv1[1]};
    textures[2] = {tri_primitive.uv2[0], tri_primitive.uv2[1]};
    normalize_uv(textures[0]);
    normalize_uv(textures[1]);
    normalize_uv(textures[2]);
  }

}  // namespace geometry

class Object3D {
 public:
  axstd::span<float> vertices;
  axstd::span<float> uv;
  axstd::span<float> colors;
  axstd::span<float> normals;
  axstd::span<float> bitangents;
  axstd::span<float> tangents;
  axstd::span<unsigned> indices;

 public:
  CLASS_DCM(Object3D)

  ax_device_callable void getTri(geometry::face_data_tri &geom, const unsigned int idx[3]) const {
    geometry::load_vertex_attribute3f(geom.v0, geom.v1, geom.v2, idx, vertices);
    geometry::load_vertex_attribute3f(geom.n0, geom.n1, geom.n2, idx, normals);
    geometry::load_vertex_attribute3f(geom.c0, geom.c1, geom.c2, idx, colors);
    geometry::load_vertex_attribute2f(geom.uv0, geom.uv1, geom.uv2, idx, uv);
    geometry::load_vertex_attribute3f(geom.tan0, geom.tan1, geom.tan2, idx, tangents);
    geometry::load_vertex_attribute3f(geom.bit0, geom.bit1, geom.bit2, idx, bitangents);
  }
};

struct IndexedTriangle {
  unsigned i0, i1, i2;
  const Object3D *mesh;
};

#endif