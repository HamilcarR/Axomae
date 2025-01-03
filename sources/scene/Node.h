#ifndef NODE_H
#define NODE_H

#include "SceneNodeInterface.h"
#include "internal/common/math/utils_3D.h"
#include "internal/datastructure/AbstractNode.h"

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
  glm::mat4 local_transformation{};       /*<Local transformation of the node*/
  glm::mat4 accumulated_transformation{}; /*<Matrix equal to all ancestors transformations*/
  int flag{};                             /*<Flag indicate what operations need to be done on this node*/
  bool ignore_transformation{false};      /*<If ignore is true , this node will not impact it's children transformations*/

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
  ax_no_discard virtual SceneTreeNode *getParent() const;
  ax_no_discard const glm::mat4 &getLocalModelMatrix() const override { return local_transformation; }
  void setLocalModelMatrix(const glm::mat4 &local_mat) override { local_transformation = local_mat; }
  void resetLocalModelMatrix() override;
  void resetAccumulatedMatrix() override;
  void setAccumulatedModelMatrix(const glm::mat4 &matrix) override { accumulated_transformation = matrix; };
  ax_no_discard const glm::mat4 &getAccumulatedModelMatrix() const override { return accumulated_transformation; };
  ax_no_discard bool isTransformIgnored() const override { return ignore_transformation; }
  void ignoreTransformation(bool ignore_) override { ignore_transformation = ignore_; }
  void reset() override;
};

/***************************************************************************************************************************************************/

// TODO: [AX-35] Implement graph nodes

#endif