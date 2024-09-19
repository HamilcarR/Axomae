#ifndef MATH_RANDOM_INTERFACE_H
#define MATH_RANDOM_INTERFACE_H

#include "math_includes.h"
#include <internal/common/axstd/span.h>
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
    int nrandi(int min, int max) { return static_cast<T *>(this)->nrandi(min, max); };
    float nrandf(float min, float max) { return static_cast<T *>(this)->nrandf(min, max); }
    glm::vec3 nrand3f(float min, float max) { return static_cast<T *>(this)->nrand3f(min, max); };
    bool randb() { return static_cast<T *>(this)->randb(); }
  };

  /* Interface for a stateless QMC random generator engine.
   * Thread safe , and usable in GPU kernels.
   * Only holds immutable configuration states, like dimensions and seeds.
   */

  template<class SUBTYPE, class FPTYPE>
  class RQMC {
   public:
    FPTYPE generate(std::size_t index, unsigned dimension) { return static_cast<SUBTYPE *>(this)->generate(index, dimension); }

    void generate(std::size_t index, FPTYPE *buffer_address, std::size_t buffer_size_bytes) {
      axstd::span<FPTYPE> buffer = {buffer_address, buffer_size_bytes / sizeof(FPTYPE)};
      generate(index, buffer);
    }

    void generate(std::size_t index, axstd::span<FPTYPE> &output_samples) {
      for (unsigned i = 0; i < output_samples.size(); i++) {
        output_samples[i] = generate(index, i);
      }
    }
  };

}  // namespace math::random
#endif  // MATH_RANDOM_INTERFACE_H
