#ifndef MESH_GEOMETRY_UTILS_H
#define MESH_GEOMETRY_UTILS_H
#include "geometry_shapes.h"
#include "internal/geometry/Object3D.h"

class Object3DBuilder {
private:
  std::vector <float> vertices ;
  std::vector <float> normals ;
  std::vector<float> colors ;
  std::vector<float> tangents ;
  std::vector<float> bitangents ;
  std::vector<unsigned int> indices ;
  std::vector<float> uv ;

 public:

  Object3DBuilder() {
  vertices = CUBE::vertices;
  normals = CUBE::normals;
  colors = CUBE::colors;
  tangents = CUBE::tangents;
  bitangents = CUBE::bitangents;
  indices = CUBE::indices;
    uv = CUBE::uv;

  }

   Object3D build_cube() {
    Object3D obj;
    obj.vertices = axstd::span<float>(vertices.data() , vertices.size());
    obj.uv = axstd::span<float>(uv.data() , uv.size());
    obj.colors = axstd::span<float>(colors.data() , colors.size());
    obj.normals = axstd::span<float>(normals.data() , normals.size());
    obj.bitangents = axstd::span<float>(bitangents.data() , bitangents.size());
    obj.tangents = axstd::span<float>(tangents.data() , tangents.size());
    obj.indices = axstd::span<unsigned>(indices.data() , indices.size());
    return obj;
  }
};

#endif  // MESH_GEOMETRY_UTILS_H
