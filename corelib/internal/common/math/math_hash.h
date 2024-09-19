#ifndef MATH_HASH_H
#define MATH_HASH_H

#include <boost/container_hash/hash.hpp>
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

}  // namespace math

#endif
