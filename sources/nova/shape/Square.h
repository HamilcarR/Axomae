#ifndef PLANE_H
#define PLANE_H
#include "math_utils.h"
#include "nova_shape.h"

namespace nova::shape {
  class Square : public NovaShapeInterface {
   private:
    glm::vec3 origin{};
    glm::vec3 side_w{};
    glm::vec3 side_h{};
    glm::vec3 normal{};

   public:
    explicit Square(const glm::vec3 &origin_) : origin(origin_) {}
    /**
     * side_w and side_h must be perpendicular.
     * Constructor will normalize the normal.
     **/
    Square(const glm::vec3 &origin_, const glm::vec3 &side_w_, const glm::vec3 &side_h_) : origin(origin_), side_w(side_w_), side_h(side_h_) {
      normal = glm::normalize(glm::cross(side_w, side_h));
    }
    ~Square() override = default;
    Square(const Square &other) = default;
    Square(Square &&other) noexcept = default;
    Square &operator=(const Square &other) = default;
    Square &operator=(Square &&other) noexcept = default;

    bool intersect(const Ray &ray, float tmin, float tmax, glm::vec3 &normal_at_intersection, float &t) const override;
  };
}  // namespace nova::shape

#endif  // PLANE_H
