#ifndef SPHERE_H
#define SPHERE_H
#include "BoundingBox.h"
#include "math_utils.h"
#include "ray/Hitable.h"

namespace nova::shape {
  class Sphere {
   private:
    float radius{};
    glm::vec3 origin{};

   public:
    Sphere() = default;
    Sphere(const glm::vec3 &origin, float radius);
    ~Sphere() = default;
    Sphere(const Sphere &copy) = delete;
    Sphere(Sphere &&move) noexcept = default;
    Sphere &operator=(const Sphere &copy) = delete;
    Sphere &operator=(Sphere &&move) noexcept = default;
    bool hit(const Ray &ray, float tmin, float tmax, hit_data &data, base_options *user_options) const;
    [[nodiscard]] glm::vec3 centroid() const { return origin; }
    [[nodiscard]] geometry::BoundingBox computeAABB() const;
  };
}  // namespace nova::shape
#endif  // SPHERE_H
