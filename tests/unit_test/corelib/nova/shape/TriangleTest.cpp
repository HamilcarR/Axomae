#include "geometry.h"
#include <Test.h>
#include <unit_test/utils/mesh_geometry_utils.h>

struct face {
  glm::vec3 v0;
  glm::vec3 v1;
  glm::vec3 v2;
};

static face get_face(const Object3D &obj) {
  const auto &indices = obj.indices;
  const auto &vertices = obj.vertices;
  unsigned idx[3] = {indices[0], indices[1], indices[2]};
  glm::vec3 v0 = {
      vertices[idx[0] * Object3D::face_stride], vertices[idx[0] * Object3D::face_stride + 1], vertices[idx[0] * Object3D::face_stride + 2]};
  glm::vec3 v1 = {
      vertices[idx[1] * Object3D::face_stride], vertices[idx[1] * Object3D::face_stride + 1], vertices[idx[1] * Object3D::face_stride + 2]};
  glm::vec3 v2 = {
      vertices[idx[2] * Object3D::face_stride], vertices[idx[2] * Object3D::face_stride + 1], vertices[idx[2] * Object3D::face_stride + 2]};

  return {v0, v1, v2};
}

TEST(TriangleTest, getTransform) {
  MeshContextBuilder ctx_builder = MeshContextBuilder(1);
  Object3DBuilder obj_builder = Object3DBuilder(Object3DBuilder::CUBE);
  ctx_builder.addMesh(obj_builder.toObject3D(), glm::mat4(1.f));
  nova::shape::MeshCtx ctx = ctx_builder.getCtx();
  nova::shape::Triangle triangle = nova::shape::Triangle(0, 0);

  nova::shape::transform::transform4x4_t transform = triangle.getTransform(ctx);
  ASSERT_EQ(transform.m, glm::mat4(1.f));
}

TEST(TriangleTest, computeAABB) {
  MeshContextBuilder builder = MeshContextBuilder(1);
  MeshContextBuilder ctx_builder = MeshContextBuilder(1);
  Object3DBuilder obj_builder = Object3DBuilder(Object3DBuilder::CUBE);
  ctx_builder.addMesh(obj_builder.toObject3D(), glm::mat4(1.f));
  nova::shape::MeshCtx ctx = ctx_builder.getCtx();
  nova::shape::Triangle triangle = nova::shape::Triangle(0, 0);

  auto aabb = triangle.computeAABB(ctx);
  Object3DBuilder compared_obj_builder = Object3DBuilder(Object3DBuilder::CUBE);
  Object3D default_cube = compared_obj_builder.build_cube();
  face f = get_face(default_cube);
  geometry::BoundingBox bbox = geometry::BoundingBox({f.v0, f.v1, f.v2});
  ASSERT_EQ(aabb, bbox);
}

TEST(TriangleTest, centroid) {
  MeshContextBuilder builder = MeshContextBuilder(1);
  MeshContextBuilder ctx_builder = MeshContextBuilder(1);
  Object3DBuilder obj_builder = Object3DBuilder(Object3DBuilder::CUBE);
  ctx_builder.addMesh(obj_builder.toObject3D(), glm::mat4(1.f));
  nova::shape::MeshCtx ctx = ctx_builder.getCtx();
  nova::shape::Triangle triangle = nova::shape::Triangle(0, 0);
  Object3D default_cube = obj_builder.build_cube();
  face f = get_face(default_cube);
  float x_center = (f.v0.x + f.v1.x + f.v2.x) / 3.f;
  float y_center = (f.v0.y + f.v1.y + f.v2.y) / 3.f;
  float z_center = (f.v0.z + f.v1.z + f.v2.z) / 3.f;
  glm::vec3 epsilon = glm::vec3(0.00001f);
  glm::vec3 test_centroid = {x_center, y_center, z_center};
  glm::vec3 centroid = triangle.centroid(ctx);
  glm::vec3 diff = glm::abs(test_centroid - centroid);
  ASSERT_LE(diff.x, epsilon.x);
  ASSERT_LE(diff.y, epsilon.y);
  ASSERT_LE(diff.z, epsilon.z);
}
