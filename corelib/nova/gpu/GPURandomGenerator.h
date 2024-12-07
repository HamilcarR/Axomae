#ifndef GPURANDOMGENERATOR_H
#define GPURANDOMGENERATOR_H
#include "internal/common/math/math_random.h"

namespace math::random {
  class GPURandomGenerator : public RandomGeneratorInterface {
   public:
    int nrandi(int min, int max) override { return 0; }
    double nrandf(double min, double max) override { return 0; }
    bool randb() override { return false; }

    CLASS_DCM(GPURandomGenerator)

    explicit GPURandomGenerator(uint64_t seed) {}
  };
}  // namespace math::random
#endif  // GPURANDOMGENERATOR_H
