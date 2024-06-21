#ifndef TRIANGLE_H
#define TRIANGLE_H
#include "BoundingBox.h"
#include "math_utils.h"
#include "nova_shape.h"

namespace nova::shape {
  class Triangle final : public NovaShapeInterface {
   private:
    glm::vec3 v0{}, v1{}, v2{};
    glm::vec3 n0{}, n1{}, n2{};
    glm::vec3 e1{};
    glm::vec3 e2{};
    glm::vec3 center{};

   public:
    Triangle() = default;
    Triangle(const glm::vec3 &v0_, const glm::vec3 &v1_, const glm::vec3 &v2_);
    Triangle(const glm::vec3 vertices[3], const glm::vec3 normals[3]);
    ~Triangle() override = default;
    Triangle(const Triangle &other) = default;
    Triangle(Triangle &&other) noexcept = default;
    Triangle &operator=(const Triangle &other) = default;
    Triangle &operator=(Triangle &&other) noexcept = default;
    bool hit(const Ray &ray, float tmin, float tmax, hit_data &data, base_options *user_options) const override;
    [[nodiscard]] glm::vec3 centroid() const override { return center; }
    [[nodiscard]] geometry::BoundingBox computeAABB() const override;

   private:
    [[nodiscard]] bool hasValidNormals() const { return n0 != glm::vec3(0.f) && n1 != glm::vec3(0.f) && n2 != glm::vec3(0.f); }
  };
}  // namespace nova::shape

#endif  // TRIANGLE_H
