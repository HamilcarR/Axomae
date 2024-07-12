#ifndef BOX_H
#define BOX_H
#include "BoundingBox.h"
#include "ShapeInterface.h"
#include <glm/vec3.hpp>

namespace nova::shape {
  class Box final : public NovaShapeInterface {
   private:
    geometry::BoundingBox aabb;

   public:
    Box() = default;
    Box(const glm::vec3 &min_coords, const glm::vec3 &max_coords);
    explicit Box(const std::vector<float> &vertices);
    explicit Box(const geometry::BoundingBox &aabb);
    ~Box() override = default;
    Box(const Box &other) = default;
    Box(Box &&other) noexcept = default;
    Box &operator=(const Box &other) = default;
    Box &operator=(Box &&other) noexcept = default;

    /**
     * @brief tmin must be initialized to a small value (0.f) while tmax should be set at a highest value.
     * Use camera near and far .
     */
    [[nodiscard]] bool hit(const Ray &ray, float tmin, float tmax, hit_data &data, base_options *user_options) const override;
    [[nodiscard]] glm::vec3 centroid() const override { return aabb.getPosition(); }
    [[nodiscard]] const glm::vec3 &getPosition() const { return aabb.getPosition(); }
    [[nodiscard]] geometry::BoundingBox computeAABB() const override { return aabb; }
  };
}  // namespace nova::shape

#endif  // BOX_H
