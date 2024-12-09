#ifndef MATH_RANDOM_H
#define MATH_RANDOM_H

#include <internal/device/gpgpu/device_utils.h>
#include <internal/macro/class_macros.h>
#include <random>

#if defined(AXOMAE_USE_CUDA)
struct kernel_argpack_t;
#endif

namespace math::random {

  template<class T>
  class AbstractRandomGenerator {
   public:
    ax_device_callable int nrandi(int min, int max) { return this->nrandi(min, max); };
    ax_device_callable double nrandf(double min, double max) { return this->nrandf(min, max); }
    ax_device_callable bool randb() { return this->randb(); }
  };

  class CPUPseudoRandomGenerator : public AbstractRandomGenerator<CPUPseudoRandomGenerator> {
   private:
    std::mt19937 m_generator;
    std::uniform_int_distribution<int> m_int_distrib;
    std::uniform_real_distribution<double> m_float_distrib;

   public:
    CPUPseudoRandomGenerator();
    explicit CPUPseudoRandomGenerator(uint64_t seed);
    int nrandi(int min, int max);
    double nrandf(double min, double max);
    bool randb();
  };

};      // namespace math::random
#endif  // math_random_H
