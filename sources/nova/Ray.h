#ifndef RAY_H
#define RAY_H
#include "math_utils.h"
namespace nova {
  class Ray {
   public:
    glm::vec3 origin;
    glm::vec3 direction;

   public:
    Ray() = default;
    Ray(const glm::vec3 &origin, const glm::vec3 &direction);
    explicit Ray(const glm::vec3 &direction);
    ~Ray() = default;
    Ray(const Ray &) = default;
    Ray(Ray &&) = default;
    Ray &operator=(const Ray &) = default;
    Ray &operator=(Ray &&) = default;

    [[nodiscard]] glm::vec3 pointAt(float t) const;
  };

}  // namespace nova
#endif
