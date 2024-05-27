#include "shape/BoundingBox.h"
#include "Test.h"
#include "math_utils.h"
#include "ray/Ray.h"
#define f_rand math::random::nrandf(-2000.f, 2000.f)
using namespace nova::shape;

constexpr float EPSILON = 0.001f;
constexpr unsigned ITERATION_NUMBER = 50;

const float MIN_COORD = -200.f;
const float MAX_COORD = 200.f;
const std::vector<float> vertices = {
    -1, -1, -1,  // 0
    1,  -1, -1,  // 1
    -1, 1,  -1,  // 2
    1,  1,  -1,  // 3
    -1, -1, 1,   // 4
    1,  -1, 1,   // 5
    -1, 1,  1,   // 6
    1,  1,  1    // 7
};

// clang-format off
const std::vector<float> directions = {
  1.f, 0.f , 0.f ,
  0.f , 1.f , 0.f ,
  0.f , 0.f , 1.f ,
  -1.f , 0.f , 0.f ,
  0.f , -1.f , 0.f ,
  0.f , 0.f , -1.f

};

// clang-format on

class AABBGenerator {
 public:
  std::array<float, 24> vertices;

 public:
  AABBGenerator() {
    // clang-format off
    vertices = {
        -MAX_COORD, -MAX_COORD, -MAX_COORD,
        MAX_COORD,  -MAX_COORD, -MAX_COORD,
        -MAX_COORD, MAX_COORD,  -MAX_COORD,
        MAX_COORD,  MAX_COORD,  -MAX_COORD,
        -MAX_COORD, -MAX_COORD, MAX_COORD,
        MAX_COORD,  -MAX_COORD, MAX_COORD,
        -MAX_COORD, MAX_COORD,  MAX_COORD,
        MAX_COORD,  MAX_COORD,  MAX_COORD
    };
    // clang-format on
  }
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

inline bool is_outside_box(const glm::vec3 &position, const BoundingBox &test) {
  return position.x < test.getMinCoords().x || position.x > test.getMaxCoords().x || position.y < test.getMinCoords().y ||
         position.y > test.getMaxCoords().y || position.z < test.getMinCoords().z || position.z > test.getMaxCoords().z;
}

TEST(BoundingBoxTest, outIntersection) {
  AABBGenerator vert_gen;
  std::vector<float> vector_vertices;
  vector_vertices.resize(vert_gen.vertices.size());
  std::copy(vert_gen.vertices.begin(), vert_gen.vertices.end(), vector_vertices.begin());
  BoundingBox B1(vector_vertices);
  for (unsigned i = 0; i < ITERATION_NUMBER; i++) {
    glm::vec3 out_position;
    bool isoutside = false;
    do {
      out_position = {f_rand, f_rand, f_rand};
      isoutside = is_outside_box(out_position, B1);
    } while (!isoutside);
    const nova::Ray ray(out_position, glm::normalize(B1.getPosition() - out_position));
    glm::vec3 n;
    float intersect_dist = 0.f;
    EXPECT_TRUE(B1.intersect(ray, 0.001, 10000.f, n, intersect_dist));
  }
}

TEST(BoundingBoxTest, inIntersection) {
  AABBGenerator vert_gen;
  std::vector<float> vector_vertices;
  vector_vertices.resize(vert_gen.vertices.size());
  std::copy(vert_gen.vertices.begin(), vert_gen.vertices.end(), vector_vertices.begin());
  BoundingBox B1(vector_vertices);
  for (unsigned i = 0; i < ITERATION_NUMBER; i++) {
    glm::vec3 out_position;
    bool isoutside = false;
    do {
      out_position = {f_rand, f_rand, f_rand};
      isoutside = is_outside_box(out_position, B1);
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