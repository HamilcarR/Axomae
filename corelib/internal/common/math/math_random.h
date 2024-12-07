#ifndef MATH_RANDOM_H
#define MATH_RANDOM_H

#include <internal/device/gpgpu/device_utils.h>
#include <internal/macro/class_macros.h>
#include <random>

#if defined(AXOMAE_USE_CUDA)
struct kernel_argpack_t;
#endif

namespace math::random {

  class RandomGeneratorInterface {
   public:
    virtual ~RandomGeneratorInterface() = default;
    virtual int nrandi(int min, int max) = 0;
    virtual double nrandf(double min, double max) = 0;
    virtual bool randb() = 0;
  };

  class CPURandomGenerator : public RandomGeneratorInterface {
   private:
    std::mt19937 m_generator;
    std::uniform_int_distribution<int> m_int_distrib;
    std::uniform_real_distribution<double> m_float_distrib;

   public:
    CPURandomGenerator();
    explicit CPURandomGenerator(uint64_t seed);
    int nrandi(int min, int max) override;
    double nrandf(double min, double max) override;
    bool randb() override;
  };

};      // namespace math::random
#endif  // math_random_H
