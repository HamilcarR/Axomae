#include "shape/BoundingBox.h"
#include "Test.h"
#include "math_utils.h"
#include "ray/Ray.h"
#define f_rand math::random::nrandf(-10.040, 12.100)
using namespace nova::shape;

constexpr float EPSILON = 0.0001f;
constexpr unsigned ITERATION_NUMBER = 5;
constexpr std::vector<float> vertices = {
    -1, -1, -1,  // 0
    1,  -1, -1,  // 1
    -1, 1,  -1,  // 2
    1,  1,  -1,  // 3
    -1, -1, 1,   // 4
    1,  -1, 1,   // 5
    -1, 1,  1,   // 6
    1,  1,  1    // 7
};

TEST(BoundingBoxTest, minMaxCompute) {
  std::vector<float> v;
  glm::vec3 max_coords = glm::vec3(-INT_MAX, -INT_MAX, -INT_MAX);
  glm::vec3 min_coords = glm::vec3(-1.f) * max_coords;
  for (unsigned i = 0; i < 24; i++)
    v.push_back(f_rand);
  for (unsigned i = 0; i < 24; i += 3) {
    max_coords.x = v[i] >= max_coords.x ? v[i] : max_coords.x;
    max_coords.y = v[i + 1] >= max_coords.y ? v[i + 1] : max_coords.y;
    max_coords.z = v[i + 2] >= max_coords.z ? v[i + 2] : max_coords.z;
    min_coords.x = v[i] < min_coords.x ? v[i] : min_coords.x;
    min_coords.y = v[i + 1] < min_coords.y ? v[i + 1] : min_coords.y;
    min_coords.z = v[i + 2] < min_coords.z ? v[i + 2] : min_coords.z;
  }
  BoundingBox B1(v);
  auto B1_max = B1.getMaxCoords();
  auto B1_min = B1.getMinCoords();
  EXPECT_EQ(B1_max, max_coords);
  EXPECT_EQ(B1_min, min_coords);
}

TEST(BoundingBoxTest, productOperatorTest) {
  BoundingBox B1(vertices);
  std::vector<std::pair<glm::mat4, glm::vec3[3]>>
      matrix_result;  // first = matrices , second = result matrix x coords => {min_coords , max_coords , center}
  matrix_result.resize(ITERATION_NUMBER);
  for (unsigned i = 0; i < ITERATION_NUMBER; i++) {
    glm::vec4 row1(f_rand, f_rand, f_rand, f_rand);
    glm::vec4 row2(f_rand, f_rand, f_rand, f_rand);
    glm::vec4 row3(f_rand, f_rand, f_rand, f_rand);
    glm::vec4 row4(f_rand, f_rand, f_rand, f_rand);
    glm::mat4 mat(row1, row2, row3, row4);
    glm::vec3 min_coords = mat * glm::vec4(B1.getMinCoords(), 1.f);
    glm::vec3 max_coords = mat * glm::vec4(B1.getMaxCoords(), 1.f);
    glm::vec3 center = mat * glm::vec4(B1.getPosition(), 1.f);
    matrix_result[i].first = mat;
    matrix_result[i].second[0] = min_coords;
    matrix_result[i].second[1] = max_coords;
    matrix_result[i].second[2] = center;
  }
  for (unsigned i = 0; i < ITERATION_NUMBER; i++) {
    BoundingBox B2 = matrix_result[i].first * B1;
    glm::vec3 min_coords_B1 = matrix_result[i].second[0];
    glm::vec3 max_coords_B1 = matrix_result[i].second[1];
    glm::vec3 center_B1 = matrix_result[i].second[2];
    EXPECT_LE(glm::abs(B2.getMinCoords().x - min_coords_B1.x), EPSILON);
    EXPECT_LE(glm::abs(B2.getMinCoords().y - min_coords_B1.y), EPSILON);
    EXPECT_LE(glm::abs(B2.getMinCoords().z - min_coords_B1.z), EPSILON);
    EXPECT_LE(glm::abs(B2.getMaxCoords().x - max_coords_B1.x), EPSILON);
    EXPECT_LE(glm::abs(B2.getMaxCoords().y - max_coords_B1.y), EPSILON);
    EXPECT_LE(glm::abs(B2.getMaxCoords().z - max_coords_B1.z), EPSILON);
    EXPECT_LE(glm::abs(B2.getPosition().x - center_B1.x), EPSILON);
    EXPECT_LE(glm::abs(B2.getPosition().y - center_B1.y), EPSILON);
    EXPECT_LE(glm::abs(B2.getPosition().z - center_B1.z), EPSILON);
  }
}

TEST(BoundingBoxTest, intersect) { BoundingBox B1(vertices); }
