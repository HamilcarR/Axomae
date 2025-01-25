#ifndef MATH_UTILS_H
#define MATH_UTILS_H

#include "math_importance_sampling.h"
#include "math_includes.h"
#include "math_spherical.h"
#include "math_texturing.h"
#include <boost/container_hash/hash.hpp>
struct Dim2 {
  unsigned width;
  unsigned height;
};
namespace math {

  inline void compute_hash_recurse(std::size_t &seed) {}

  template<class T, class... Args>
  void compute_hash_recurse(std::size_t &seed, const T &head, const Args &...args) {
    boost::hash_combine(seed, head);
    compute_hash_recurse(seed, std::forward<Args>(args)...);
  }

  /**
   * Computes hash for v , stores it in seed and returns it.
   */
  template<class... Args>
  std::size_t hash(std::size_t &seed, const Args &...args) {
    compute_hash_recurse(seed, std::forward<const Args &>(args)...);
    return seed;
  }

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