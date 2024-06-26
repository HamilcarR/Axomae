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

namespace math {
  inline constexpr double epsilon = 1e-6;
}
#endif