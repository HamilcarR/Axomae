#ifndef TRIANGLE_H
#define TRIANGLE_H
#include "BoundingBox.h"
#include "ShapeInterface.h"
#include "math_utils.h"
#include "project_macros.h"

namespace nova::shape {

  /* Not cache efficient , for now.
   * leave this optimization for later.
   */
  class Triangle final : public NovaShapeInterface {
   private:
    glm::vec3 e1{};
    glm::vec3 e2{};
    glm::vec3 center{};
    glm::vec3 v0{}, v1{}, v2{};
    glm::vec3 n0{}, n1{}, n2{};
    glm::vec3 t0{}, t1{}, t2{};
    glm::vec3 b0{}, b1{}, b2{};
    glm::vec2 uv0{}, uv1{}, uv2{};
    bool uv_valid{false};

   public:
    CLASS_OCM(Triangle)

    explicit Triangle(const glm::vec3 vertices[3],
                      const glm::vec3 normals[3] = nullptr,
                      const glm::vec2 textures[3] = nullptr,
                      const glm::vec3 tangents[3] = nullptr,
                      const glm::vec3 bitangents[3] = nullptr);
    bool hit(const Ray &ray, float tmin, float tmax, hit_data &data, base_options *user_options) const override;
    [[nodiscard]] glm::vec3 centroid() const override { return center; }
    [[nodiscard]] geometry::BoundingBox computeAABB() const override;

    [[nodiscard]] bool hasValidTangents() const { return t0 != glm::vec3(0.f) || t1 != glm::vec3(0.f) || t2 != glm::vec3(0.f); }
    [[nodiscard]] bool hasValidBitangents() const { return b0 != glm::vec3(0.f) || b1 != glm::vec3(0.f) || b2 != glm::vec3(0.f); }
    [[nodiscard]] bool hasValidNormals() const { return n0 != glm::vec3(0.f) || n1 != glm::vec3(0.f) || n2 != glm::vec3(0.f); }
    [[nodiscard]] bool hasValidUvs() const { return uv_valid; }
  };
}  // namespace nova::shape

#endif  // TRIANGLE_H
