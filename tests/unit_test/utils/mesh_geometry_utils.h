#ifndef MESH_GEOMETRY_UTILS_H
#define MESH_GEOMETRY_UTILS_H
#include "geometry_shapes.h"
#include <internal/geometry/Object3D.h>

class Object3DBuilder {
 public:
  enum TYPE { CUBE, QUAD, TRIANGLE };

 private:
  std::vector<float> vertices;
  std::vector<float> normals;
  std::vector<float> colors;
  std::vector<float> tangents;
  std::vector<float> bitangents;
  std::vector<unsigned int> indices;
  std::vector<float> uv;

 public:
  void fillShape(TYPE type) {
    switch (type) {
      case CUBE:
        vertices = CUBE::vertices;
        normals = CUBE::normals;
        colors = CUBE::colors;
        tangents = CUBE::tangents;
        bitangents = CUBE::bitangents;
        indices = CUBE::indices;
        uv = CUBE::uv;
        break;

      case QUAD:
        vertices = QUAD::vertices;
        normals = QUAD::normals;
        colors = QUAD::colors;
        tangents = QUAD::tangents;
        bitangents = QUAD::bitangents;
        indices = QUAD::indices;
        uv = QUAD::uv;
        break;

      case TRIANGLE:
        vertices = TRIANGLE::vertices;
        normals = TRIANGLE::normals;
        colors = TRIANGLE::colors;
        tangents = TRIANGLE::tangents;
        bitangents = TRIANGLE::bitangents;
        indices = TRIANGLE::indices;
        uv = TRIANGLE::uv;
        break;
    }
  }

  Object3D toObject3D() {
    Object3D obj;
    obj.vertices = axstd::span<float>(vertices);
    obj.uv = axstd::span<float>(uv);
    obj.colors = axstd::span<float>(colors);
    obj.normals = axstd::span<float>(normals);
    obj.bitangents = axstd::span<float>(bitangents);
    obj.tangents = axstd::span<float>(tangents);
    obj.indices = axstd::span<unsigned>(indices);
    return obj;
  }

  Object3DBuilder(TYPE type = CUBE) { fillShape(type); }

  Object3D build_cube() {
    fillShape(CUBE);
    return toObject3D();
  }

  Object3D build_quad() {
    fillShape(QUAD);
    return toObject3D();
  }

  Object3D build_triangle() {
    fillShape(TRIANGLE);
    return toObject3D();
  }
};

#endif  // MESH_GEOMETRY_UTILS_H
