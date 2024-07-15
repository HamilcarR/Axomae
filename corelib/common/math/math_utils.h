#ifndef MATH_UTILS_H
#define MATH_UTILS_H
#define GLM_ENABLE_EXPERIMENTAL
#include "math_importance_sampling.h"
#include "math_random.h"
#include "math_spherical.h"
#include "math_texturing.h"
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/vec3.hpp>

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