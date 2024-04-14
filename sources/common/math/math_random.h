#ifndef MATH_RANDOM_H
#define MATH_RANDOM_H
#include <random>

namespace math::random {

  inline std::mt19937 init_rand() {
    std::random_device rd;
    std::mt19937 gen(rd());
    return gen;
  }

  inline std::uniform_int_distribution<int> getUniformIntDistrib(int min, int max) { return std::uniform_int_distribution<int>(min, max); }

  inline std::uniform_real_distribution<double> getUniformDoubleDistrib(double min, double max) {
    return std::uniform_real_distribution<double>(min, max);
  }

  inline int nrandi(int n1, int n2) {
    auto gen = init_rand();
    auto distrib = getUniformIntDistrib(n1, n2);
    return distrib(gen);
  }

  inline double nrandf(double n1, double n2) {
    auto gen = init_rand();
    auto distrib = getUniformDoubleDistrib(n1, n2);
    return distrib(gen);
  }

  inline bool randb() { return nrandi(0, 1); }
};      // namespace math::random
#endif  // math_random_H