#ifndef AABB_UTILS_TEST_H
#define AABB_UTILS_TEST_H
#include "geometry_shapes.h"
#include "internal/common/math/math_utils.h"
#include <internal/common/math/math_random.h>
constexpr float EPSILON = 0.001f;
constexpr unsigned ITERATION_NUMBER = 50;

const float MIN_COORD = -200.f;
const float MAX_COORD = 200.f;

inline float f_rand(math::random::CPUPseudoRandomGenerator &generator) { return (float)generator.nrandf(-2000.f, 2000.f); }

const std::vector<float> vertices = CUBE::vertices;

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
