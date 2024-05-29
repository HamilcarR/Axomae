#ifndef NODE_H
#define NODE_H

#include "AbstractNode.h"
#include "SceneNodeInterface.h"
#include "utils_3D.h"

/**
 * @file Node.h
 * Implements the node class in the scene
 */

/**
 * @class SceneTreeNode
 * @brief Provides implementation for a scene tree node
 */
class SceneTreeNode : public SceneNodeInterface, public datastructure::AbstractNode {
 protected:
  int flag{};                             /*<Flag indicate what operations need to be done on this node*/
  glm::mat4 local_transformation{};       /*<Local transformation of the node*/
  glm::mat4 accumulated_transformation{}; /*<Matrix equal to all ancestors transformations*/

 protected:
  explicit SceneTreeNode(SceneTreeNode *parent = nullptr, datastructure::AbstractHierarchy *owner = nullptr);
  SceneTreeNode(const std::string &name,
                const glm::mat4 &transformation,
                SceneTreeNode *parent = nullptr,
                datastructure::AbstractHierarchy *owner = nullptr);

 public:
  ~SceneTreeNode() override = default;
  SceneTreeNode(const SceneTreeNode &copy) = default;
  SceneTreeNode(SceneTreeNode &&move) noexcept = default;
  SceneTreeNode &operator=(const SceneTreeNode &copy) = default;
  SceneTreeNode &operator=(SceneTreeNode &&move) noexcept = default;

  /**
   * Computes the final transformation of the node based on the transformation of it's parent.
   */
  glm::mat4 computeFinalTransformation() override;
  virtual void setParent(NodeInterface *node);
  [[nodiscard]] virtual SceneTreeNode *getParent() const;
  [[nodiscard]] const glm::mat4 &getLocalModelMatrix() const override { return local_transformation; }
  void setLocalModelMatrix(const glm::mat4 &local_mat) override { local_transformation = local_mat; }
  void resetLocalModelMatrix() override;
  void resetAccumulatedMatrix() override;
  void setAccumulatedModelMatrix(const glm::mat4 &matrix) override { accumulated_transformation = matrix; };
  [[nodiscard]] const glm::mat4 &getAccumulatedModelMatrix() const override { return accumulated_transformation; };

  void reset() override;
};

/***************************************************************************************************************************************************/

// TODO: [AX-35] Implement graph nodes

#endif