#ifndef SPHERE_H
#define SPHERE_H
#include "math_utils.h"
#include "nova_shape.h"
#include "scene/Hitable.h"

namespace nova::shape {
  class Sphere final : public NovaShapeInterface {
   private:
    float radius{};
    glm::vec3 origin{};

   public:
    Sphere() = default;
    Sphere(const glm::vec3 &origin, float radius);
    ~Sphere() override = default;
    Sphere(const Sphere &copy) = delete;
    Sphere(Sphere &&move) noexcept = default;
    Sphere &operator=(const Sphere &copy) = delete;
    Sphere &operator=(Sphere &&move) noexcept = default;
    bool intersect(const Ray &ray, float tmin, float tmax, glm::vec3 &normal, float &t) const override;
  };
}  // namespace nova::shape
#endif  // SPHERE_H
