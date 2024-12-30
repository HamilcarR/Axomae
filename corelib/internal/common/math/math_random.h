#ifndef MATH_RANDOM_H
#define MATH_RANDOM_H
#include "math_includes.h"
#include "math_random_interface.h"
#include <memory>
namespace math::random {

  class CPUPseudoRandomGenerator : public AbstractRandomGenerator<CPUPseudoRandomGenerator> {
    class Impl;
    std::shared_ptr<Impl> pimpl{};

   public:
    CPUPseudoRandomGenerator();
    explicit CPUPseudoRandomGenerator(uint64_t seed);
    int nrandi(int min, int max);
    float nrandf(float min, float max);
    glm::vec3 nrand3f(float min, float max);
    bool randb();
  };

  class CPUQuasiRandomGenerator : public AbstractRandomGenerator<CPUQuasiRandomGenerator> {
   private:
    class Impl;
    std::shared_ptr<Impl> pimpl{};

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
