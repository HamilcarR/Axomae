#ifndef TRIANGLE_H
#define TRIANGLE_H
#include "math_utils.h"
#include "nova_shape.h"

namespace nova::shape {
  class Triangle final : public NovaShapeInterface {
    glm::vec3 v0{}, v1{}, v2{};
    glm::vec3 e1{};
    glm::vec3 e2{};

   public:
    Triangle() = default;
    Triangle(const glm::vec3 &v0_, const glm::vec3 &v1_, const glm::vec3 &v2_);
    ~Triangle() override = default;
    Triangle(const Triangle &other) = default;
    Triangle(Triangle &&other) noexcept = default;
    Triangle &operator=(const Triangle &other) = default;
    Triangle &operator=(Triangle &&other) noexcept = default;

    bool intersect(const Ray &ray, float tmin, float tmax, glm::vec3 &normal_at_intersection, float &t) const override;
  };
}  // namespace nova::shape

#endif  // TRIANGLE_H
