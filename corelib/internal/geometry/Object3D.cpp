#include "Object3D.h"
#include "project_macros.h"
void Object3D::get_tri(geometry::face_data_tri &geom, const unsigned int idx[3]) const {
  geometry::load_vertex_attribute3f(geom.v0, geom.v1, geom.v2, idx, vertices);
  geometry::load_vertex_attribute3f(geom.n0, geom.n1, geom.n2, idx, normals);
  geometry::load_vertex_attribute3f(geom.c0, geom.c1, geom.c2, idx, colors);
  geometry::load_vertex_attribute2f(geom.uv0, geom.uv1, geom.uv2, idx, uv);
  geometry::load_vertex_attribute3f(geom.tan0, geom.tan1, geom.tan2, idx, tangents);
  geometry::load_vertex_attribute3f(geom.bit0, geom.bit1, geom.bit2, idx, bitangents);
}
