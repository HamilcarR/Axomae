#ifndef RAY_H
#define RAY_H
#include "Vector.h"
namespace nova {
  class Ray {
   public:
    Vec3f origin;
    Vec3f direction;

   public:
    Ray() = default;
    Ray(Vec3f origin, Vec3f direction);
    explicit Ray(Vec3f direction);
    ~Ray() = default;
    Ray(const Ray &) = default;
    Ray(Ray &&) = default;
    Ray &operator=(const Ray &) = default;
    Ray &operator=(Ray &&) = default;

    [[nodiscard]] Vec3f pointAt(float t) const;
  };

}  // namespace nova
#endif
