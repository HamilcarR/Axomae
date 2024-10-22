#ifndef BOX_H
#define BOX_H
#include "internal/geometry/BoundingBox.h"
#include "ray/Hitable.h"
#include "ray/Ray.h"

namespace nova::shape {
  class Box {
   private:
    geometry::BoundingBox aabb;

   public:
    Box() = default;
    Box(const glm::vec3 &min_coords, const glm::vec3 &max_coords);
    explicit Box(const std::vector<float> &vertices);
    explicit Box(const geometry::BoundingBox &aabb);
    ~Box() = default;
    Box(const Box &other) = default;
    Box(Box &&other) noexcept = default;
    Box &operator=(const Box &other) = default;
    Box &operator=(Box &&other) noexcept = default;

    /**
     * @brief tmin must be initialized to a small value (0.f) while tmax should be set at a highest value.
     * Use camera near and far .
     */
    ax_no_discard bool hit(const Ray &ray, float tmin, float tmax, hit_data &data, base_options *user_options) const;
    ax_no_discard glm::vec3 centroid() const { return aabb.getPosition(); }
    ax_no_discard const glm::vec3 &getPosition() const { return aabb.getPosition(); }
    ax_no_discard geometry::BoundingBox computeAABB() const { return aabb; }
  };
}  // namespace nova::shape

#endif  // BOX_H
