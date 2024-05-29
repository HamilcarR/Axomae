#include "BoundingBox.h"
#include "Test.h"
#include "geometry/aabb_utils_test.h"
#include "ray/Ray.h"
#include "shape/Box.h"

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
  for (unsigned i = 0; i < ITERATION_NUMBER; i++) {
    glm::vec3 out_position;
    bool isoutside = false;
    do {
      out_position = {f_rand, f_rand, f_rand};
      isoutside = is_outside_box(out_position, B1.computeAABB());
    } while (!isoutside);
    const nova::Ray ray(out_position, glm::normalize(B1.getPosition() - out_position));
    glm::vec3 n;
    float intersect_dist = 0.f;
    EXPECT_TRUE(B1.intersect(ray, 0.001, 10000.f, n, intersect_dist));
  }
}

TEST(BoxTest, inIntersection) {
  AABBGenerator vert_gen;
  std::vector<float> vector_vertices;
  vector_vertices.resize(vert_gen.vertices.size());
  std::copy(vert_gen.vertices.begin(), vert_gen.vertices.end(), vector_vertices.begin());
  Box B1(vector_vertices);
  for (unsigned i = 0; i < ITERATION_NUMBER; i++) {
    glm::vec3 out_position;
    bool isoutside = false;
    do {
      out_position = {f_rand, f_rand, f_rand};
      isoutside = is_outside_box(out_position, B1.computeAABB());
    } while (isoutside);
    for (int i = 0; i < directions.size(); i += 3) {
      const glm::vec3 direction{directions[i], directions[i + 1], directions[i + 2]};
      const nova::Ray ray(out_position, direction);
      glm::vec3 n;
      float intersect_dist = 0.f;
      EXPECT_TRUE(B1.intersect(ray, 0.001, 10000.f, n, intersect_dist));
    }
  }
}
