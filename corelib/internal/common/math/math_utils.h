#ifndef MATH_UTILS_H
#define MATH_UTILS_H

#include "math_importance_sampling.h"
#include "math_includes.h"
#include "math_spherical.h"
#include "math_texturing.h"

struct Dim2 {
  unsigned width;
  unsigned height;
};
namespace math {
  inline constexpr double epsilon = 1e-6;
  namespace calculus {

    /* Will compute X in :
     * N = 1 + 2 + 3 + 4 + 5 + ... + X
     */
    template<class T>
    float compute_serie_term(T N) {
      return (-1 + std::sqrt(1.f + 8 * N)) * 0.5f;
    }

  }  // namespace calculus

}  // namespace math

#endif