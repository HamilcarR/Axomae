#ifndef MESH_GEOMETRY_UTILS_H
#define MESH_GEOMETRY_UTILS_H
#include "Object3D.h"
#include "geometry_shapes.h"

class Object3DBuilder {
 public:
  static Object3D build_cube() {
    Object3D obj;
    obj.vertices = CUBE::vertices;
    obj.uv = CUBE::uv;
    obj.colors = CUBE::colors;
    obj.normals = CUBE::normals;
    obj.bitangents = CUBE::bitangents;
    obj.tangents = CUBE::tangents;
    obj.indices = CUBE::indices;
    return obj;
  }
};

#endif  // MESH_GEOMETRY_UTILS_H
