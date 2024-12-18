#ifndef MATH_RANDOM_H
#define MATH_RANDOM_H

#include <boost/random/sobol.hpp>
#include <boost/random/uniform_01.hpp>
#include <glm/vec3.hpp>
#include <internal/device/gpgpu/device_utils.h>
#include <internal/macro/class_macros.h>
#include <internal/macro/project_macros.h>
#include <random>
#if defined(AXOMAE_USE_CUDA)
struct kernel_argpack_t;
#endif

namespace math::random {

  /* For rand a float or double in  [0,1] , returns [min , max] range */
  template<class T, class U>
  constexpr T to_interval(T min, T max, U rand) {
    static_assert(std::is_integral_v<T> || std::is_floating_point_v<T>);
    return min + (rand * (max - min));
  }

  template<class T>
  class AbstractRandomGenerator {
   public:
    ax_device_callable int nrandi(int min, int max) { return static_cast<T *>(this)->nrandi(min, max); };
    ax_device_callable float nrandf(float min, float max) { return static_cast<T *>(this)->nrandf(min, max); }
    ax_device_callable glm::vec3 nrand3f(float min, float max) { return static_cast<T *>(this)->nrand3f(min, max); };
    ax_device_callable bool randb() { return static_cast<T *>(this)->randb(); }
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
    float nrandf(float min, float max);
    glm::vec3 nrand3f(float min, float max);
    bool randb();
  };

  class CPUQuasiRandomGenerator : public AbstractRandomGenerator<CPUQuasiRandomGenerator> {
   private:
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

};      // namespace math::random
#endif  // math_random_H
