#ifndef MATH_IMPORTANCE_SAMPLING_H
#define MATH_IMPORTANCE_SAMPLING_H
#include "math_includes.h"


namespace math::importance_sampling {
  inline glm::vec3 pgc3d(unsigned x, unsigned y, unsigned z) {
    x = x * 1664525u + 1013904223u;
    y = y * 1664525u + 1013904223u;
    z = z * 1664525u + 1013904223u;
    x += y * z;
    y += z * x;
    z += x * y;
    x ^= x >> 16u;
    y ^= y >> 16u;
    z ^= z >> 16u;
    x += y * z;
    y += z * x;
    z += x * y;
    return glm::vec3(x, y, z) * (1.f / float(0xFFFFFFFFu));
  }

  inline double radicalInverse(unsigned bits) {
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return double(bits) * 2.3283064365386963e-10;
  }

  inline glm::dvec3 hammersley3D(unsigned i, unsigned N) {
    return glm::dvec3(double(i) / double(N), radicalInverse(i), radicalInverse(i ^ 0xAAAAAAAAu));
  }

}  // namespace math::importance_sampling
#endif  // math_importance_sampling_H
