#include "math_random.h"
#include <boost/random/sobol.hpp>
#include <boost/random/uniform_01.hpp>
#include <internal/macro/class_macros.h>
#include <internal/macro/project_macros.h>

#include <random>

namespace math::random {

  class CPUPseudoRandomGenerator::Impl {
   private:
    std::mt19937 m_generator;
    std::uniform_int_distribution<int> m_int_distrib;
    std::uniform_real_distribution<double> m_float_distrib;

   public:
    Impl() {
      std::random_device random_device{};
      m_generator = std::mt19937{random_device()};
    }
    Impl(uint64_t seed) { m_generator = std::mt19937{seed}; }

    decltype(auto) operator()(int min, int max) {
      std::uniform_int_distribution<int>::param_type dist(min, max);
      m_int_distrib.param(dist);
      return m_int_distrib(m_generator);
    }
    template<class REALTYPE>
    REALTYPE operator()(REALTYPE min, REALTYPE max) {
      std::uniform_real_distribution<double>::param_type dist(min, max);
      m_float_distrib.param(dist);
      return static_cast<REALTYPE>(m_float_distrib(m_generator));
    }
  };

  CPUPseudoRandomGenerator::CPUPseudoRandomGenerator() { pimpl = std::make_shared<Impl>(); }

  CPUPseudoRandomGenerator::CPUPseudoRandomGenerator(uint64_t seed) { pimpl = std::make_shared<Impl>(seed); }

  int CPUPseudoRandomGenerator::nrandi(int min, int max) {
    if (min > max)
      std::swap(min, max);
    return (*pimpl)(min, max);
  }

  float CPUPseudoRandomGenerator::nrandf(float min, float max) {
    if (min > max)
      std::swap(min, max);
    return (*pimpl)(min, max);
  }

  glm::vec3 CPUPseudoRandomGenerator::nrand3f(float min, float max) {
    if (min > max)
      std::swap(min, max);
    return {(*pimpl)(min, max), (*pimpl)(min, max), (*pimpl)(min, max)};
  }

  bool CPUPseudoRandomGenerator::randb() { return nrandi(0, 1); }

  /*****************************************************************************************************************************************************/

  class CPUQuasiRandomGenerator::Impl {
   private:
    boost::random::sobol sobol_engine{1};
    boost::random::uniform_01<> dist;

   public:
    Impl(uint64_t seed, uint64_t dimension) {
      sobol_engine = boost::random::sobol(dimension);
      sobol_engine.seed(seed);
    }
    decltype(auto) operator()() { return dist(sobol_engine); }
  };

  CPUQuasiRandomGenerator::CPUQuasiRandomGenerator() { pimpl = std::make_shared<Impl>(0xDEADBEEF, 1); }

  CPUQuasiRandomGenerator::CPUQuasiRandomGenerator(uint64_t seed, uint64_t dimension) { pimpl = std::make_shared<Impl>(seed, dimension); }

  int CPUQuasiRandomGenerator::nrandi(int min, int max) {
    if (min > max)
      std::swap(min, max);
    return to_interval(min, max, (*pimpl)());
  }
  float CPUQuasiRandomGenerator::nrandf(float min, float max) {
    if (min > max)
      std::swap(min, max);
    return to_interval(min, max, (*pimpl)());
  }

  glm::vec3 CPUQuasiRandomGenerator::nrand3f(float min, float max) {
    if (min > max)
      std::swap(min, max);
    return {to_interval(min, max, (*pimpl)()), to_interval(min, max, (*pimpl)()), to_interval(min, max, (*pimpl)())};
  }

  bool CPUQuasiRandomGenerator::randb() { return nrandi(0, 1) == 1; }

}  // namespace math::random