#ifndef NODE_H
#define NODE_H

#include "NodeInterface.h"
#include "SceneNodeInterface.h"
#include "utils_3D.h"
#include <memory>
/**
 * @file Node.h
 * Implements the node class in the scene
 */

/**
 * @class SceneTreeNode
 * @brief Provides implementation for a scene tree node
 */
class SceneTreeNode : public SceneNodeInterface, public NodeInterface {
 protected:
  bool mark;                              /*<Generic mark , for graph traversal*/
  bool updated;                           /*<Lets the owning structure know if node has been modified */
  std::string name;                       /*<Name of the node*/
  std::vector<NodeInterface *> parents;   /*<List of parents*/
  std::vector<NodeInterface *> children;  /*<List of children*/
  AbstractHierarchy *owner;               /*<Structure owning this hierarchy*/
  int flag;                               /*<Flag indicate what operations need to be done on this node*/
  glm::mat4 local_transformation{};       /*<Local transformation of the node*/
  glm::mat4 accumulated_transformation{}; /*<Matrix equal to all ancestors transformations*/

 protected:
  explicit SceneTreeNode(SceneTreeNode *parent = nullptr, AbstractHierarchy *owner = nullptr);
  SceneTreeNode(const std::string &name, const glm::mat4 &transformation, SceneTreeNode *parent = nullptr, AbstractHierarchy *owner = nullptr);

 public:
  ~SceneTreeNode() override = default;
  SceneTreeNode(const SceneTreeNode &copy) = default;
  SceneTreeNode(SceneTreeNode &&move) noexcept = default;
  SceneTreeNode &operator=(const SceneTreeNode &copy) = default;
  SceneTreeNode &operator=(SceneTreeNode &&move) noexcept = default;

 public:
  glm::mat4 computeFinalTransformation() override;
  virtual void setParent(NodeInterface *node);
  [[nodiscard]] virtual SceneTreeNode *getParent() const;
  [[nodiscard]] const std::vector<NodeInterface *> &getChildren() const override { return children; };
  [[nodiscard]] const std::vector<NodeInterface *> &getParents() const override { return parents; }
  void setParents(std::vector<NodeInterface *> &parents) override;
  void addChildNode(NodeInterface *node) override;
  void emptyParents() override;
  void emptyChildren() override;
  [[nodiscard]] bool isLeaf() const override;
  void setMark(bool marked) override { mark = marked; };
  [[nodiscard]] bool isMarked() const override { return mark; };
  void setName(const std::string &str) override { name = str; };
  [[nodiscard]] const std::string &getName() const override { return name; };
  void setHierarchyOwner(AbstractHierarchy *_owner) override { owner = _owner; };
  [[nodiscard]] AbstractHierarchy *getHierarchyOwner() const override { return owner; };
  [[nodiscard]] bool isRoot() const override;
  NodeInterface *returnRoot() override;
  [[nodiscard]] bool isUpdated() const override { return updated; }
  void reset() override;

  [[nodiscard]] const glm::mat4 &getLocalModelMatrix() const override { return local_transformation; }
  void setLocalModelMatrix(const glm::mat4 &local_mat) override { local_transformation = local_mat; }
  void resetLocalModelMatrix() override;
  void resetAccumulatedMatrix() override;
  void setAccumulatedModelMatrix(const glm::mat4 &matrix) override { accumulated_transformation = matrix; };
  [[nodiscard]] const glm::mat4 &getAccumulatedModelMatrix() const override { return accumulated_transformation; };
};

/***************************************************************************************************************************************************/

// TODO: [AX-35] Implement graph nodes

#endif