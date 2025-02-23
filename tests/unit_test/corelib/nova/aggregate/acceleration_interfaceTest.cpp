#include "aggregate/acceleration_interface.h"
#include "aggregate/aggregate_datastructures.h"
#include "geometry.h"
#include "glm/ext/matrix_transform.hpp"
#include "primitive/NovaGeoPrimitive.h"
#include "ray/Ray.h"
#include "shape/Triangle.h"
#include <internal/common/math/math_camera.h>
#include <internal/common/math/math_random.h>
#include <internal/common/math/utils_3D.h>
#include <internal/geometry/Object3D.h>
#include <unit_test/Test.h>

PrimitiveAggregateBuilder::PrimitiveAggregateBuilder(const glm::mat4 &transform) : transform_storage(false) {
  Object3D mesh = mesh_builder.toObject3D();
  std::size_t size_primit = mesh.indices.size() / Object3D::face_stride;
  triangles_collection.reserve(size_primit);
  shapes_collection.reserve(size_primit);
  geometric_primitives.reserve(size_primit);
  primitives_collection.reserve(size_primit);
  for (std::size_t index = 0; index < mesh.indices.size(); index += Object3D::face_stride) {
    nova::shape::Triangle tri = {0, index};

    triangles_collection.push_back(tri);
    shapes_collection.push_back(&triangles_collection.back());
    geometric_primitives.push_back(nova::primitive::NovaGeoPrimitive(shapes_collection.back(), {}));
    primitives_collection.push_back(&geometric_primitives.back());
  }
  mesh_list.push_back(mesh);
  transform_storage.init(1);  // We work with one mesh for now , pointing to offset 0 of the transform array
  transform_storage.add(transform, 0);
}

nova::aggregate::primitive_aggregate_data_s PrimitiveAggregateBuilder::generateAggregate() {
  nova::aggregate::primitive_aggregate_data_s aggregate{};
  aggregate.primitive_list_view = {primitives_collection};
  aggregate.mesh_geometry.transforms = transform_storage.getTransformViews();
  aggregate.mesh_geometry.geometry.host_geometry_view = {mesh_list};
  return aggregate;
}

/**********************************************************************************************************************************************************************************/
static nova::aggregate::GenericAccelerator<nova::aggregate::NativeBuild> build_acceleration_structure(PrimitiveAggregateBuilder &builder) {
  nova::aggregate::GenericAccelerator<nova::aggregate::NativeBuild> bvh;
  bvh.build(builder.generateAggregate());
  return bvh;
}

static bvh_hit_data gen_hit_data() {
  constexpr static bool is_rendering = true;
  bvh_hit_data hit_data{};
  hit_data.is_rendering = &is_rendering;
  return hit_data;
}

/* d is a direction vector. */
inline nova::Ray gen_ray(const glm::vec3 &o, const glm::vec3 &d) {
  nova::Ray ray{};
  ray.origin = o;
  ray.direction = d;
  return ray;
}
/* center is a point in world space. */
inline glm::mat4 get_inv_view(const glm::vec3 &eye, const glm::vec3 &center) {
  return glm::inverse(glm::lookAt(eye, center, glm::vec3(0.f, 1.f, 0.f)));
}

inline nova::Ray gen_ray(const glm::mat4 &inv_v, float dx, float dy) {
  math::camera::camera_ray c = math::camera::ray_inv_mat(dx, dy, glm::mat4(1.f), inv_v);
  return {c.near, c.far};
}
/**
 * When within a cube , should always intersect at least one face with a random direction.
 */
TEST(acceleration_test, intersect_inside) {
  glm::mat4 transform = glm::mat4(1.f);
  PrimitiveAggregateBuilder builder{transform};
  auto bvh = build_acceleration_structure(builder);
  math::random::CPUPseudoRandomGenerator generator;
  for (int i = 0; i < 100; i++) {
    bvh_hit_data hit = gen_hit_data();
    glm::vec3 dir = generator.nrand3f(-2.f, 2.f);
    const nova::Ray ray = gen_ray({0.f, 0.f, 0.f}, {dir.x, dir.y, dir.z});
    EXPECT_TRUE(bvh.hit(ray, hit));
  }
}

/**
 * When outside a cube , should intersect at least one face if it is in front of the camera.
 */
TEST(acceleration_test, intersect_outside_front) {
  glm::mat4 transform = glm::mat4(1.f);
  PrimitiveAggregateBuilder builder{transform};
  auto bvh = build_acceleration_structure(builder);
  math::random::CPUPseudoRandomGenerator generator;
  for (int i = 0; i < 100; i++) {
    bvh_hit_data hit = gen_hit_data();
    glm::vec3 dir = generator.nrand3f(-1.f, 1.f);
    nova::Ray ray = gen_ray(get_inv_view({0.f, 0.f, -2.f}, {0.f, 0.f, -1.f}), dir.x, dir.y);
    EXPECT_TRUE(bvh.hit(ray, hit));
  }
}

/**
 * When outside a cube , should not intersect any face behind the camera.
 */
TEST(acceleration_test, nointersect_outside_back) {
  glm::mat4 transform = glm::mat4(1.f);
  PrimitiveAggregateBuilder builder{transform};
  auto bvh = build_acceleration_structure(builder);
  math::random::CPUPseudoRandomGenerator generator;
  for (int i = 0; i < 100; i++) {
    bvh_hit_data hit = gen_hit_data();
    glm::vec3 dir = generator.nrand3f(-1.f, 1.f);
    nova::Ray ray = gen_ray(get_inv_view({0.f, 0.f, -1.0001f}, {0.f, 0.f, -2.f}), dir.x, dir.y);
    EXPECT_FALSE(bvh.hit(ray, hit));
  }
}

/**
 * When outside a cube , should not intersect any face when cube is translated 3 units to the right.
 */
TEST(acceleration_test, nointersect_outside_back_translated) {
  glm::mat4 transform = math::geometry::construct_transformation_matrix({1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {3, 0, 0});
  PrimitiveAggregateBuilder builder{transform};
  auto bvh = build_acceleration_structure(builder);
  math::random::CPUPseudoRandomGenerator generator;
  for (int i = 0; i < 100; i++) {
    bvh_hit_data hit = gen_hit_data();
    glm::vec3 dir = generator.nrand3f(-1.f, 1.f);
    const nova::Ray ray = gen_ray(get_inv_view({0.f, 0.f, -1.f}, {0.f, 0.f, 0.f}), dir.x, dir.y);
    EXPECT_FALSE(bvh.hit(ray, hit));
  }
}
