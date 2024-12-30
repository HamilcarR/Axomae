#ifndef MATH_RANDOM_H
#define MATH_RANDOM_H
#include "math_includes.h"
#include "math_random_interface.h"
#include <boost/random/sobol.hpp>
#include <boost/random/uniform_01.hpp>
#include <internal/macro/class_macros.h>
#include <internal/macro/project_macros.h>
#include <random>

namespace math::random {

  class CPUPseudoRandomGenerator : public AbstractRandomGenerator<CPUPseudoRandomGenerator> {
    std::mt19937 m_generator;
    std::uniform_int_distribution<int> m_int_distrib;
    std::uniform_real_distribution<double> m_float_distrib;

   public:
    CPUPseudoRandomGenerator();
    explicit CPUPseudoRandomGenerator(uint64_t seed);
    int nrandi(int min, int max);
    float nrandf(float min, float max);
    glm::vec3 nrand3f(float min, float max);
    bool randb();
  };

  class CPUQuasiRandomGenerator : public AbstractRandomGenerator<CPUQuasiRandomGenerator> {
    boost::random::sobol sobol_engine{1};
    boost::random::uniform_01<> dist;

   public:
    CPUQuasiRandomGenerator();
    CPUQuasiRandomGenerator(uint64_t seed, uint64_t dimension);
    int nrandi(int min, int max);
    float nrandf(float min, float max);
    glm::vec3 nrand3f(float min, float max);
    bool randb();
  };

};  // namespace math::random
#endif  // math_random_H
