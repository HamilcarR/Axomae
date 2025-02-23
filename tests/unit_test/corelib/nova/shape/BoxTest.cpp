#include "shape/Box.h"
#include "ray/Ray.h"
#include <internal/geometry/BoundingBox.h>
#include <unit_test/Test.h>
#include <unit_test/utils/aabb_utils_test.h>

using namespace geometry;
using namespace nova::shape;

inline bool is_outside_box(const glm::vec3 &position, const BoundingBox &test) {
  return position.x < test.getMinCoords().x || position.x > test.getMaxCoords().x || position.y < test.getMinCoords().y ||
         position.y > test.getMaxCoords().y || position.z < test.getMinCoords().z || position.z > test.getMaxCoords().z;
}

TEST(BoxTest, outIntersection) {
  AABBGenerator vert_gen;
  std::vector<float> vector_vertices;
  vector_vertices.resize(vert_gen.vertices.size());
  std::copy(vert_gen.vertices.begin(), vert_gen.vertices.end(), vector_vertices.begin());
  Box B1(vector_vertices);
  math::random::CPUPseudoRandomGenerator generator;
  for (unsigned i = 0; i < ITERATION_NUMBER; i++) {
    glm::vec3 out_position;
    bool isoutside = false;
    do {
      out_position = {f_rand(generator), f_rand(generator), f_rand(generator)};
      isoutside = is_outside_box(out_position, B1.computeAABB());
    } while (!isoutside);
    const nova::Ray ray(out_position, glm::normalize(B1.getPosition() - out_position));
    nova::hit_data data;
    EXPECT_TRUE(B1.hit(ray, 0.001, 10000.f, data, {}));
  }
}

TEST(BoxTest, inIntersection) {
  AABBGenerator vert_gen;
  std::vector<float> vector_vertices;
  vector_vertices.resize(vert_gen.vertices.size());
  std::copy(vert_gen.vertices.begin(), vert_gen.vertices.end(), vector_vertices.begin());
  Box B1(vector_vertices);
  math::random::CPUPseudoRandomGenerator generator;
  for (unsigned i = 0; i < ITERATION_NUMBER; i++) {
    glm::vec3 out_position;
    bool isoutside = false;
    do {
      out_position = {f_rand(generator), f_rand(generator), f_rand(generator)};
      isoutside = is_outside_box(out_position, B1.computeAABB());
    } while (isoutside);
    for (int i = 0; i < directions.size(); i += 3) {
      const glm::vec3 direction{directions[i], directions[i + 1], directions[i + 2]};
      const nova::Ray ray(out_position, direction);
      glm::vec3 n;
      nova::hit_data data;
      EXPECT_TRUE(B1.hit(ray, 0.001, 10000.f, data, {}));
    }
  }
}
