#ifndef BOUNDINGBOX_H
#define BOUNDINGBOX_H
#include "math_utils.h"
#include "nova_shape.h"

/**
 * @brief File implementing an OBB calculations
 * @file BoundingBox.h
 */

namespace nova::shape {
  /* Replace Inheritance from SceneTreeNode by SceneNodeInterface*/
  class BoundingBox final : public NovaShapeInterface {

    friend BoundingBox operator*(const glm::mat4 &matrix, const BoundingBox &bounding_box) {
      glm::vec3 min_c = matrix * glm::vec4(bounding_box.getMinCoords(), 1.f);
      glm::vec3 max_c = matrix * glm::vec4(bounding_box.getMaxCoords(), 1.f);
      return {min_c, max_c};
    }

   private:
    glm::vec3 max_coords;
    glm::vec3 min_coords;
    glm::vec3 center;

   public:
    BoundingBox();
    explicit BoundingBox(const std::vector<float> &geometry);
    BoundingBox(const glm::vec3 &min_coords, const glm::vec3 &max_coords);
    ~BoundingBox() override = default;
    BoundingBox(const BoundingBox &copy);
    BoundingBox(BoundingBox &&move) noexcept;
    BoundingBox &operator=(const BoundingBox &copy);
    BoundingBox &operator=(BoundingBox &&move) noexcept;
    [[nodiscard]] virtual const glm::vec3 &getPosition() const { return center; }
    /**
     * @brief Compute the position of the AABB in view space.
     * @param modelview Modelview matrix : Model x View
     * @return glm::vec4 Position of the AABB relative to the camera
     */
    [[nodiscard]] virtual glm::vec3 computeModelViewPosition(const glm::mat4 &modelview) const;
    /**
     * @brief Returns the index + vertices array representatives of the bounding box
     * @return std::pair<std::vector<float> , std::vector<unsigned>>
     */
    [[nodiscard]] virtual std::pair<std::vector<float>, std::vector<unsigned>> getVertexArray() const;
    [[nodiscard]] const glm::vec3 &getMaxCoords() const { return max_coords; }
    [[nodiscard]] const glm::vec3 &getMinCoords() const { return min_coords; }
    void setMaxCoords(glm::vec3 max) { max_coords = max; }
    void setMinCoords(glm::vec3 min) { min_coords = min; }
    /**
     * @brief tmin must be initialized to a small value (0.f) while tmax should be set at a highest value.
     * Use camera near and far .
     */
    [[nodiscard]] bool intersect(const Ray &ray, float tmin, float tmax, glm::vec3 &normal_at_intersection, float &t) const override;
  };
}  // namespace nova::shape
#endif
