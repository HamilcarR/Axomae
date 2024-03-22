#ifndef BOUNDINGBOX_H
#define BOUNDINGBOX_H
#include "Node.h"
#include "utils_3D.h"

/**
 * @brief File implementing the bounding boxes calculations
 * @file BoundingBox.h
 */

class BoundingBox : public SceneTreeNode {

  friend BoundingBox operator*(const glm::mat4 &matrix, const BoundingBox &bounding_box);

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
};

#endif
