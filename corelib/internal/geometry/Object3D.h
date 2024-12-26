#ifndef OBJECT3D_H
#define OBJECT3D_H
#include "internal/macro/project_macros.h"
#include <glm/glm.hpp>
#include <internal/common/axstd/span.h>
#include <vector>

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
  static void load_vertex_attribute3f(T c0[3], T c1[3], T c2[3], const unsigned idx[3], const axstd::span<T> &attribute) {
    if (attribute.empty())
      return;
    for (int i = 0; i < 3; i++) {
      c0[i] = attribute[idx[0] * 3 + i];
      c1[i] = attribute[idx[1] * 3 + i];
      c2[i] = attribute[idx[2] * 3 + i];
    }
  }

  template<class T>
  static void load_vertex_attribute2f(T c0[2], T c1[2], T c2[2], const unsigned idx[3], const axstd::span<T> &attribute) {
    if (attribute.empty())
      return;
    for (int i = 0; i < 2; i++) {
      c0[i] = attribute[idx[0] * 2 + i];
      c1[i] = attribute[idx[1] * 2 + i];
      c2[i] = attribute[idx[2] * 2 + i];
    }
  }

  void transform_vertices(const face_data_tri &tri_primitive, const glm::mat4 &final_transfo, glm::vec3 vertices[3]);
  void transform_normals(const face_data_tri &tri_primitive, const glm::mat3 &normal_matrix, glm::vec3 normals[3]);
  void transform_tangents(const face_data_tri &tri_primitive, const glm::mat3 &normal_matrix, glm::vec3 tangents[3]);
  void transform_bitangents(const face_data_tri &tri_primitive, const glm::mat3 &normal_matrix, glm::vec3 bitangents[3]);
  void extract_uvs(const face_data_tri &tri_primitive, glm::vec2 textures[3]);


}  // namespace geometry

class Object3D {
 public:
  axstd::span<float> vertices;
  axstd::span<float> uv;
  axstd::span<float> colors;
  axstd::span<float> normals;
  axstd::span<float> bitangents;
  axstd::span<float> tangents;
  axstd::span<unsigned int> indices;

 public:
  CLASS_CM(Object3D)

  void getTri(geometry::face_data_tri &geom, const unsigned int indices[3]) const;

};

#endif