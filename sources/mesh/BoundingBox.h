#ifndef BOUNDINGBOX_H
#define BOUNDINGBOX_H
#include "Node.h"
#include "sources/common/math/utils_3D.h"

/**
 * @brief File implementing the bounding boxes calculations
 * @file BoundingBox.h
 */

/**
 * @brief Class implementing bounding boxes computation
 * @class BoundingBox
 *
 */
class BoundingBox : public SceneTreeNode {
 public:
  /**
   * @brief Construct a new Bounding Box object
   *
   */
  BoundingBox();

  /**
   * @brief Construct a new Bounding Box object
   *
   * @param geometry
   */
  BoundingBox(const std::vector<float> &geometry);

  /**
   * @brief Construct a new Bounding Box object
   *
   * @param min_coords (X , Y , Z) with each components as the minimal coordinate of the box
   * @param max_coords (X , Y , Z) with each components as the maximal coordinates of the box
   */
  BoundingBox(glm::vec3 min_coords, glm::vec3 max_coords);

  /**
   * @brief Destroy the Bounding Box object
   *
   */
  virtual ~BoundingBox();

  /**
   * @brief Product operator between a 4x4 matrix , and a BoundingBox object.
   *
   * @param matrix Matrix that will be multiplied by the current bounding box
   * @param bounding_box The bounding box we want to apply a transformation on.
   * @return BoundingBox A new Bounding box with transformed coordinates and center using the matrix argument
   */
  friend BoundingBox operator*(const glm::mat4 &matrix, const BoundingBox &bounding_box);

  /**
   * @brief Get the Position of the AABB
   *
   * @return glm::vec3 Center of the AABB
   */
  virtual glm::vec3 getPosition() const { return center; }

  /**
   * @brief Compute the position of the AABB in view space.
   *
   * @param modelview Modelview matrix : Model x View
   * @return glm::vec4 Position of the AABB relative to the camera
   */
  virtual glm::vec3 computeModelViewPosition(glm::mat4 modelview) const { return glm::vec3(modelview * glm::vec4(center, 1.f)); }

  /**
   * @brief Returns the index + vertices array representatives of the bounding box
   *
   * @return std::pair<std::vector<float> , std::vector<unsigned>>
   */
  virtual std::pair<std::vector<float>, std::vector<unsigned>> getVertexArray() const;

  /**
   * @brief Get the Max Coords object
   *
   * @return glm::vec3
   */
  glm::vec3 getMaxCoords() const { return max_coords; }

  /**
   * @brief Get the Min Coords object
   *
   * @return glm::vec3
   */
  glm::vec3 getMinCoords() const { return min_coords; }

  /**
   * @brief Set the Max Coords object
   *
   * @param max
   */
  void setMaxCoords(glm::vec3 max) { max_coords = max; }

  /**
   * @brief Set the Min Coords object
   *
   * @param min
   */
  void setMinCoords(glm::vec3 min) { min_coords = min; }

 private:
  glm::vec3 max_coords;
  glm::vec3 min_coords;
  glm::vec3 center;
};

#endif
