#include "Ray.h"

namespace nova {

  Ray::Ray(const glm::vec3 &o, const glm::vec3 &d) : origin(o), direction(d) {}
  Ray::Ray(const glm::vec3 &d) : origin(0), direction(d) {}
  glm::vec3 Ray::pointAt(float t) const { return origin + t * direction; }
}  // namespace nova