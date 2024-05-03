#ifndef SPHERE_H
#define SPHERE_H
#include "Hitable.h"
#include "math_utils.h"

namespace nova {
  class Sphere : public Hitable {
   private:
    float radius;
    glm::vec3 origin;

   public:
    Sphere() = default;
    Sphere(const glm::vec3 &origin, float radius);
    ~Sphere() override = default;
    Sphere(const Sphere &copy) = default;
    Sphere(Sphere &&move) noexcept = default;
    Sphere &operator=(const Sphere &copy) = default;
    Sphere &operator=(Sphere &&move) noexcept = default;
    bool hit(const Ray &r, float tmin, float tmax, hit_data &data, const base_options *user_options) const override;
  };
}  // namespace nova
#endif  // SPHERE_H
