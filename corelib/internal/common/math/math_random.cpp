#include "math_random.h"

namespace math::random {

  CPUPseudoRandomGenerator::CPUPseudoRandomGenerator() {
    std::random_device rd{};
    m_generator = std::mt19937(rd());
  }

  CPUPseudoRandomGenerator::CPUPseudoRandomGenerator(uint64_t seed) { m_generator = std::mt19937(seed); }

  int CPUPseudoRandomGenerator::nrandi(int min, int max) {
    if (min > max)
      std::swap(min, max);
    std::uniform_int_distribution<int>::param_type dist(min, max);
    m_int_distrib.param(dist);
    return m_int_distrib(m_generator);
  }

  double CPUPseudoRandomGenerator::nrandf(double min, double max) {
    if (min > max)
      std::swap(min, max);
    std::uniform_real_distribution<double>::param_type dist(min, max);
    m_float_distrib.param(dist);
    return m_float_distrib(m_generator);
  }

  bool CPUPseudoRandomGenerator::randb() { return nrandi(0, 1); }

  CPUQuasiRandomGenerator::CPUQuasiRandomGenerator() {
    sobol_engine.seed(0xDEADBEEF);
    sobol_engine = boost::random::sobol(1);
  }

  CPUQuasiRandomGenerator::CPUQuasiRandomGenerator(uint64_t seed, uint64_t dimension) {
    sobol_engine = boost::random::sobol(dimension);
    sobol_engine.seed(seed);
  }

  int CPUQuasiRandomGenerator::nrandi(int min, int max) {
    if (min > max)
      std::swap(min, max);
    return to_interval(min, max, dist(sobol_engine));
  }

  double CPUQuasiRandomGenerator::nrandf(double min, double max) {
    if (min > max)
      std::swap(min, max);
    return to_interval(min, max, dist(sobol_engine));
  }

  bool CPUQuasiRandomGenerator::randb() { return nrandi(0, 1) == 1; }

}  // namespace math::random