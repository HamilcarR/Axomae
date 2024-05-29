#ifndef AABB_UTILS_TEST_H
#define AABB_UTILS_TEST_H

#include "math_utils.h"
#define f_rand math::random::nrandf(-2000.f, 2000.f)

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
#endif  // AABB_UTILS_TEST_H
