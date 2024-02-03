#ifndef NODE_H
#define NODE_H
#include "Logger.h"
#include "sources/common/math/utils_3D.h"
#include <memory>

/**
 * @file Node.h
 * Implements the node class in the scene
 *
 */

class ISceneHierarchy;

/**
 * @class INode
 * @brief Generic graph node. Can be used for any graph-like structures and processing , including scene manipulations
 */
class INode {
 public:
  enum FLAG : signed { EMPTY = 0 };

  virtual ~INode() = default;

  /**
   * @brief Returns the array of children
   *
   * @return const std::vector<INode*>&
   */
  virtual const std::vector<INode *> &getChildren() const = 0;

  /**
   * @brief Returns the array of parents
   *
   * @return const std::vector<ISceneNode*>&
   */
  virtual const std::vector<INode *> &getParents() const = 0;

  /**
   * @brief Add a child in the array of children
   *
   * @param node
   */
  virtual void addChildNode(INode *node) = 0;

  /**
   * @brief Set up a new array of parents
   *
   * @param parents
   */
  virtual void setParents(std::vector<INode *> &parents) = 0;

  /**
   * @brief Empty up the list of parents , setting the size of property "parents" to 0
   *
   */
  virtual void emptyParents();

  /**
   * @brief Empty up the list of children
   *
   */
  virtual void emptyChildren();

  /**
   * @brief Check if present node doesn't have successors
   *
   */
  virtual bool isLeaf() const;

  /**
   * @brief Mark the current node
   *
   * @param marked
   */
  void setMark(bool marked) { mark = marked; }

  /**
   * @brief Check if the current node is marked
   *
   */
  bool isMarked() { return mark; }

  /**
   * @brief Set the Name of the node
   *
   * @param str
   */
  void setName(std::string str) { name = str; }

  /**
   * @brief Get the Name of the node
   *
   * @return const std::string&
   */
  const std::string &getName() const { return name; }

  /**
   * @brief Specify the structure that owns this node
   *
   * @param _owner
   */
  void setHierarchyOwner(ISceneHierarchy *_owner) { owner = _owner; }

  /**
   * @brief Returns a pointer on the structure that owns this node
   *
   * @return SceneHierarchyInterface*
   */
  ISceneHierarchy *getHierarchyOwner() const { return owner; }

  /**
   * @brief Check if node is root
   *
   * @return true
   * @return false
   */
  virtual bool isRoot() const;

  /**
   * @brief If node isn't root , will travel the hierarchy and returns the root of the hierarchy
   *
   * @return ISceneNode*
   */
  virtual INode *returnRoot() = 0;

  /**
   * @brief Checks if the node has been updated
   *
   * @return true
   */
  virtual bool isUpdated() const { return updated; }

  /**
   * @brief Provides cleaning of various resources , API , IO , etc
   *
   */
  virtual void clean();

 protected:
  bool mark;                     /*<Generic mark , for graph traversal*/
  bool updated;                  /*<Lets the owning structure know if node has been modified */
  std::string name;              /*<Name of the node*/
  std::vector<INode *> parents;  /*<List of parents*/
  std::vector<INode *> children; /*<List of children*/
  ISceneHierarchy *owner;        /*<Structure owning this hierarchy*/
  int flag;                      /*<Flag indicate what operations need to be done on this node*/
};

/**
 * @class ISceneNode
 * @brief Provides an interface for a scene node
 */
class ISceneNode : public INode {
 public:
  /**
   * @brief Compute the final model matrix , by multiplying the accumulated transformation with the local transformation
   *
   * @return glm::mat4
   */
  virtual glm::mat4 computeFinalTransformation() = 0;

  /**
   * @brief Get the Local Model Matrix
   *
   * @return const glm::mat4
   */
  virtual const glm::mat4 getLocalModelMatrix() const { return local_transformation; }

  /**
   * @brief Set the Local Model Matrix object
   *
   * @param matrix
   */
  virtual void setLocalModelMatrix(const glm::mat4 matrix) { local_transformation = matrix; }

  /**
   * @brief Sets the local transformation to identity
   *
   */
  virtual void resetLocalModelMatrix();

  virtual void resetAccumulatedMatrix();
  /**
   * @brief Set the Accumulated Model Matrix
   *
   * @param matrix
   */
  virtual void setAccumulatedModelMatrix(const glm::mat4 &matrix) { accumulated_transformation = matrix; }

  /**
   * @brief Get the Accumulated Model Matrix
   *
   * @return const glm::mat4&
   */
  virtual const glm::mat4 &getAccumulatedModelMatrix() const { return accumulated_transformation; }

  virtual void clean();

 protected:
  glm::mat4 local_transformation;       /*<Local transformation of the node*/
  glm::mat4 accumulated_transformation; /*<Matrix equal to all ancestors transformations*/

  // TODO: [AX-37] Replace with adjacency list + pointer on one node that is a parent
};

/***************************************************************************************************************************************************/
/**
 * @class SceneTreeNode
 * @brief Provides implementation for a scene tree node
 */
class SceneTreeNode : public ISceneNode {
 protected:
  /**
   * @brief Construct a new Scene Tree Node object
   *
   * @param parent Predecessor node in the scene hierarchy
   * @param owner The structure that owns this node
   */
  explicit SceneTreeNode(ISceneNode *parent = nullptr, ISceneHierarchy *owner = nullptr);

  /**
   * @brief Construct a new Scene Tree Node object
   *
   * @param name
   * @param transformation
   * @param parent
   * @param owner
   */
  SceneTreeNode(const std::string &name, const glm::mat4 &transformation, ISceneNode *parent = nullptr, ISceneHierarchy *owner = nullptr);

  /**
   * @brief Set the predecessor of this node in the scene tree . Note that only the first element of the vector is
   * stored as parent , as this structure is a tree.
   * @param parents Vector of predecessors . Only the first element is considered
   */
  void setParents(std::vector<INode *> &parents) override;

 public:
  /**
   * @brief Construct a new Scene Tree Node object
   *
   * @param copy SceneTreeNode copy
   */
  SceneTreeNode(const SceneTreeNode &copy);

  /**
   * @brief Computes the final model matrix of the object in world space
   *
   * @return glm::mat4 Model matrix in world space
   */
  virtual glm::mat4 computeFinalTransformation() override;

  /**
   * @brief Assignment operator for the SceneTreeNode
   *
   * @param copy Copy to be assigned
   * @return SceneTreeNode& Returns *this object
   */
  virtual SceneTreeNode &operator=(const SceneTreeNode &copy);

  /**
   * @brief Add a child to the present node
   *
   * @param node
   */
  virtual void addChildNode(INode *node) override;

  /**
   * @brief Set the parent of this node.
   *
   * @param node
   */
  virtual void setParent(INode *node);

  /**
   * @brief Get the Parent of this node
   *
   * @return ISceneNode*
   */
  virtual SceneTreeNode *getParent() const;

  /**
   * @brief Get the children nodes collection of this present node
   *
   * @return const std::vector<ISceneNode*>&
   */
  virtual const std::vector<INode *> &getChildren() const { return children; };

  /**
   * @brief Get the parents of this node . In case this is a SceneTreeNode , the returned vector is of size 1
   *
   * @return std::vector<ISceneNode*>
   */
  virtual const std::vector<INode *> &getParents() const { return parents; }

  /**
   * @brief Returns the root node of the tree
   *
   * @return ISceneNode* Root node
   */
  virtual ISceneNode *returnRoot();

  virtual void clean();

 protected:
};

/***************************************************************************************************************************************************/

// TODO: [AX-35] Implement graph nodes

#endif