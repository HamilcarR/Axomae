#ifndef MATH_RANDOM_INTERFACE_H
#define MATH_RANDOM_INTERFACE_H
#include "math_includes.h"

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
}  // namespace math::random
#endif  // MATH_RANDOM_INTERFACE_H
