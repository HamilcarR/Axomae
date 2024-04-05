#include "Ray.h"

namespace nova {

  Ray::Ray(Vec3f o, Vec3f d) : origin(o), direction(d) {}
  Ray::Ray(Vec3f d) : origin(0), direction(d) {}
  Vec3f Ray::pointAt(float t) const { return origin * t + direction; }
}  // namespace nova