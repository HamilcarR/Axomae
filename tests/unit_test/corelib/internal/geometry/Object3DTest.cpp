#include <internal/geometry/Object3D.h>
#include <unit_test/Test.h>
#include <unit_test/utils/mesh_geometry_utils.h>

template<class T>
static bool eq_type3(const T comp[3], const T templ[3]) {
  return comp[0] == templ[0] && comp[1] == templ[1] && comp[2] == templ[2];
}
template<class T>
static bool eq_type2(const T comp[2], const T templ[2]) {
  return comp[0] == templ[0] && comp[1] == templ[1];
}

TEST(Object3DTest, get_tri_vertices) {
  Object3DBuilder builder;
  Object3D cube = builder.build_cube();
  geometry::face_data_tri triangle{};

  for (int i = 0; i < CUBE::indices.size(); i += 3) {

    unsigned int idx[3];
    idx[0] = CUBE::indices[i];
    idx[1] = CUBE::indices[i + 1];
    idx[2] = CUBE::indices[i + 2];

    cube.getTri(triangle, idx);
    const std::vector<float> &vertices = CUBE::vertices;
    float v0[3], v1[3], v2[3];
    for (int i = 0; i < 3; i++) {
      v0[i] = vertices[idx[0] * 3 + i];
      v1[i] = vertices[idx[1] * 3 + i];
      v2[i] = vertices[idx[2] * 3 + i];
      ASSERT_EQ(v0[i], triangle.v0[i]);
      ASSERT_EQ(v1[i], triangle.v1[i]);
      ASSERT_EQ(v2[i], triangle.v2[i]);
    }
  }
}

TEST(Object3DTest, get_tri_normals) {
  Object3DBuilder builder;
  Object3D cube = builder.build_cube();
  geometry::face_data_tri triangle{};

  for (int i = 0; i < CUBE::indices.size(); i += 3) {

    unsigned int idx[3];
    idx[0] = CUBE::indices[i];
    idx[1] = CUBE::indices[i + 1];
    idx[2] = CUBE::indices[i + 2];

    cube.getTri(triangle, idx);

    float n0[3], n1[3], n2[3];
    const std::vector<float> &normals = CUBE::normals;
    for (int i = 0; i < 3; i++) {
      n0[i] = normals[idx[0] * 3 + i];
      n1[i] = normals[idx[1] * 3 + i];
      n2[i] = normals[idx[2] * 3 + i];
      ASSERT_EQ(n0[i], triangle.n0[i]);
      ASSERT_EQ(n1[i], triangle.n1[i]);
      ASSERT_EQ(n2[i], triangle.n2[i]);
    }
  }
}

TEST(Object3DTest, get_tri_colors) {
  Object3DBuilder builder;
  Object3D cube = builder.build_cube();
  geometry::face_data_tri triangle{};

  for (int i = 0; i < CUBE::indices.size(); i += 3) {

    unsigned int idx[3];
    idx[0] = CUBE::indices[i];
    idx[1] = CUBE::indices[i + 1];
    idx[2] = CUBE::indices[i + 2];

    cube.getTri(triangle, idx);

    float c0[3], c1[3], c2[3];
    const std::vector<float> &colors = CUBE::colors;
    for (int i = 0; i < 3; i++) {
      c0[i] = colors[idx[0] * 3 + i];
      c1[i] = colors[idx[1] * 3 + i];
      c2[i] = colors[idx[2] * 3 + i];
      ASSERT_EQ(c0[i], triangle.c0[i]);
      ASSERT_EQ(c1[i], triangle.c1[i]);
      ASSERT_EQ(c2[i], triangle.c2[i]);
    }
  }
}

TEST(Object3DTest, get_tri_uv) {
  Object3DBuilder builder;
  Object3D cube = builder.build_cube();
  geometry::face_data_tri triangle{};

  for (int i = 0; i < CUBE::indices.size(); i += 3) {

    unsigned int idx[3];
    idx[0] = CUBE::indices[i];
    idx[1] = CUBE::indices[i + 1];
    idx[2] = CUBE::indices[i + 2];

    cube.getTri(triangle, idx);

    float uv0[2], uv1[2], uv2[2];
    const std::vector<float> &uv = CUBE::uv;
    for (int i = 0; i < 2; i++) {
      uv0[i] = uv[idx[0] * 2 + i];
      uv1[i] = uv[idx[1] * 2 + i];
      uv2[i] = uv[idx[2] * 2 + i];
      ASSERT_EQ(uv0[i], triangle.uv0[i]);
      ASSERT_EQ(uv1[i], triangle.uv1[i]);
      ASSERT_EQ(uv2[i], triangle.uv2[i]);
    }
  }
}

TEST(Object3DTest, get_tri_tangent) {
  Object3DBuilder builder;
  Object3D cube = builder.build_cube();
  geometry::face_data_tri triangle{};

  for (int i = 0; i < CUBE::indices.size(); i += 3) {

    unsigned int idx[3];
    idx[0] = CUBE::indices[i];
    idx[1] = CUBE::indices[i + 1];
    idx[2] = CUBE::indices[i + 2];

    cube.getTri(triangle, idx);

    float tan0[3], tan1[3], tan2[3];
    const std::vector<float> &tangent = CUBE::tangents;
    for (int i = 0; i < 3; i++) {
      tan0[i] = tangent[idx[0] * 3 + i];
      tan1[i] = tangent[idx[1] * 3 + i];
      tan2[i] = tangent[idx[2] * 3 + i];
      ASSERT_EQ(tan0[i], triangle.tan0[i]);
      ASSERT_EQ(tan1[i], triangle.tan1[i]);
      ASSERT_EQ(tan2[i], triangle.tan2[i]);
    }
  }
}

TEST(Object3DTest, get_tri_bitangent) {
  Object3DBuilder builder;
  Object3D cube = builder.build_cube();
  geometry::face_data_tri triangle{};

  for (int i = 0; i < CUBE::indices.size(); i += 3) {

    unsigned int idx[3];
    idx[0] = CUBE::indices[i];
    idx[1] = CUBE::indices[i + 1];
    idx[2] = CUBE::indices[i + 2];

    cube.getTri(triangle, idx);

    float bit0[3], bit1[3], bit2[3];
    const std::vector<float> &bitangents = CUBE::bitangents;
    for (int i = 0; i < 3; i++) {
      bit0[i] = bitangents[idx[0] * 3 + i];
      bit1[i] = bitangents[idx[1] * 3 + i];
      bit2[i] = bitangents[idx[2] * 3 + i];
      ASSERT_EQ(bit0[i], triangle.bit0[i]);
      ASSERT_EQ(bit1[i], triangle.bit1[i]);
      ASSERT_EQ(bit2[i], triangle.bit2[i]);
    }
  }
}
