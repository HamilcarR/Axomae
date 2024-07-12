#ifndef OBJECT3D_H
#define OBJECT3D_H
#include "project_macros.h"
#include <cstring>
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
  static void load_vertex_attribute3f(T c0[3], T c1[3], T c2[3], const unsigned idx[3], const std::vector<T> &attribute) {
    if (attribute.empty())
      return;
    for (int i = 0; i < 3; i++) {
      c0[i] = attribute[idx[0] * 3 + i];
      c1[i] = attribute[idx[1] * 3 + i];
      c2[i] = attribute[idx[2] * 3 + i];
    }
  }

  template<class T>
  static void load_vertex_attribute2f(T c0[2], T c1[2], T c2[2], const unsigned idx[3], const std::vector<T> &attribute) {
    if (attribute.empty())
      return;
    for (int i = 0; i < 2; i++) {
      c0[i] = attribute[idx[0] * 2 + i];
      c1[i] = attribute[idx[1] * 2 + i];
      c2[i] = attribute[idx[2] * 2 + i];
    }
  }
}  // namespace geometry

/* Contains vertices data (a lot) */
class Object3D {
 public:
  std::vector<float> vertices;       /*<Vertices array*/
  std::vector<float> uv;             /*<UV arrays of dimension 2*/
  std::vector<float> colors;         /*<Colors array , Format is RGB*/
  std::vector<float> normals;        /*<Normals of the geometry*/
  std::vector<float> bitangents;     /*<Bitangent of each vertex*/
  std::vector<float> tangents;       /*<Tangent of each vertex*/
  std::vector<unsigned int> indices; /*<Indices of the vertices buffer*/

 public:
  CLASS_CM(Object3D)

  void get_tri(geometry::face_data_tri &geom, const unsigned int indices[3]) const;

  void clean() {
    vertices.clear();
    uv.clear();
    colors.clear();
    normals.clear();
    bitangents.clear();
    tangents.clear();
    indices.clear();
  }
};

#endif