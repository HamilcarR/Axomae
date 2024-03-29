#ifndef SCENEHIERARCHY_H
#define SCENEHIERARCHY_H
#include "Node.h"
#include "RenderingDatabaseInterface.h"
#include <functional>
/**
 * @file SceneGraph.h
 * @brief This file implements the hierarchy system of a scene.
 *
 */

/**
 * @class ISceneHierarchy
 *
 */
class ISceneHierarchy {
 public:
  /**
   * @brief Set the Root object
   *
   * @param _root
   */
  void setRoot(ISceneNode *_root) { root = _root; }

  /**
   * @brief Returns the iterator of the structure
   *
   * @return SceneNodeInterface*
   */
  INode *getIterator() const { return iterator; }

  /**
   * @brief Set the Iterator of the structure
   *
   * @param iter Iterator of type SceneNodeInterface*
   */
  void setIterator(INode *iter) { iterator = iter; }

  /**
   * @brief Create an empty root node , and stores it into the provided database.
   *
   */
  virtual void createGenericRootNode(IResourceDB<int, INode> &database) = 0;

  /**
   * @brief Get the Root Node object
   *
   * @return SceneNodeInterface*
   */
  INode *getRootNode() const { return root; }

  /**
   * @brief Set the node as a child of root
   *
   * @param node
   */
  void setAsRootChild(INode *node);

  /**
   * @brief This method will update the hierarchy given a list of nodes. Only the descendants of these nodes will be
   * updated.
   */
  virtual void updatedHierarchy() = 0;

  /**
   * @brief Traverse the scene structure, and set their owner of each node to this structure
   *
   */
  virtual void updateOwner();

  /**
   * @brief Traverse the scene structure , and recomputes all transformations up to the accumulated_modelmatrix of each
   * node
   *
   */
  virtual void updateAccumulatedTransformations() = 0;

  /**
   * @brief Free all generic nodes allocated and only generic nodes .
   * Other types of nodes need to be deallocated by their own structures and owners. For example , lights should be
   * freed by their LightDatabase , even though they are stored in the scene tree as well .
   *
   */
  virtual void clear();

  /**
   * @brief Returns a collection of nodes of the specified name
   *
   * @param name String the method searches for
   * @return std::vector<SceneNodeInterface*> Collection of nodes returned . Empty if the method didn't find any node of
   * specified name.
   */
  virtual std::vector<INode *> findByName(const std::string &name) = 0;

  /**
   * @brief This method will traverse the structure doing a depth first search, and apply a functor to each node.
   *
   * @tparam Args Templated arguments to pass to a functor.
   * @param begin Node from which we begin the traversal .
   * @param func Functor to be applied on each node .
   * @param args Variadic arguments of func
   */
  template<class F, class... Args>
  void dfs(INode *begin, F func, Args &&...args);

  template<class F, class... Args>
  void dfsConst(const INode *begin, F func, Args &&...args) const;

  template<class F, class... Args>
  void bfs(INode *begin, F func, Args &&...args);

  template<class F, class... Args>
  void bfsConst(const INode *begin, F func, Args &&...args) const;

 private:
  /**
   * @brief Recursive traversal of dfs.
   * @see  template<class F , class ...Args> void dfs(SceneNodeInterface* begin ,F func, Args&& ...args)
   * @tparam Args Templated arguments to pass to a functor.
   * @param begin Node from which we begin the traversal .
   * @param func Functor to be applied on each node .
   * @param args Variadic arguments of func
   */
  template<class F, class... Args>
  void dfsTraverse(INode *node, F func, Args &&...args);

  template<class F, class... Args>
  void dfsConstTraverse(const INode *node, F func, Args &&...args) const;

  template<class F, class... Args>
  void bfsTraverse(INode *node, F func, Args &&...args);

  template<class F, class... Args>
  void bfsConstTraverse(const INode *node, F func, Args &&...args) const;

 protected:
  INode *root; /*<Root of the hierarchy*/
 private:
  mutable INode *iterator; /*<Iterator to keep track of nodes in traversals*/
  mutable const INode *const_iterator;
};

/*******************************************************************************************************************************************************************/

class SceneTree : public ISceneHierarchy {

 public:
  SceneTree(ISceneNode *root = nullptr);
  SceneTree(const SceneTree &copy);
  virtual ~SceneTree();
  void createGenericRootNode(IResourceDB<int, INode> &database) override;
  virtual SceneTree &operator=(const SceneTree &copy);
  void updateAccumulatedTransformations() override;
  virtual void pushNewRoot(INode *new_root);
  std::vector<INode *> findByName(const std::string &name) override;
  void updatedHierarchy() override;

 private:
  bool node_updated;
};

/*Templates definitions*/
/*******************************************************************************************************************************************************************/

template<class F, class... Args>
void ISceneHierarchy::dfs(INode *begin, F func, Args &&...args) {
  if ((iterator = begin) != nullptr) {
    dfsTraverse(iterator, func, std::forward<Args>(args)...);
  }
}

template<class F, class... Args>
void ISceneHierarchy::dfsTraverse(INode *node, F func, Args &&...args) {
  if (node != nullptr) {
    func(node, std::forward<Args>(args)...);
    for (INode *child : node->getChildren())
      dfsTraverse(child, func, std::forward<Args>(args)...);
  }
}

template<class F, class... Args>
void ISceneHierarchy::dfsConst(const INode *begin, F func, Args &&...args) const {
  if ((const_iterator = begin) != nullptr) {
    dfsTraverse(const_iterator, func, std::forward<Args>(args)...);
  }
}

template<class F, class... Args>
void ISceneHierarchy::dfsConstTraverse(const INode *node, F func, Args &&...args) const {
  if (node != nullptr) {
    func(node, std::forward<Args>(args)...);
    for (const INode *child : node->getChildren())
      dfsTraverse(child, func, std::forward<Args>(args)...);
  }
}

template<class F, class... Args>
void ISceneHierarchy::bfs(INode *begin, F func, Args &&...args) {
  if ((iterator = begin) != nullptr) {
    bfsTraverse(iterator, func, std::forward<Args>(args)...);
  }
}

template<class F, class... Args>
void ISceneHierarchy::bfsTraverse(INode *node, F func, Args &&...args) {
  if (node != nullptr) {
    if (node->isRoot())
      func(node, std::forward<Args>(args)...);
    for (INode *child : node->getChildren())
      func(child, std::forward<Args>(args)...);
    for (INode *child : node->getChildren())
      bfsTraverse(child, func, std::forward<Args>(args)...);
  }
}

template<class F, class... Args>
void ISceneHierarchy::bfsConst(const INode *begin, F func, Args &&...args) const {
  if ((const_iterator = begin) != nullptr) {
    bfsTraverse(const_iterator, func, std::forward<Args>(args)...);
  }
}

template<class F, class... Args>
void ISceneHierarchy::bfsConstTraverse(const INode *node, F func, Args &&...args) const {
  if (node != nullptr) {
    if (node->isRoot())
      func(node, std::forward<Args>(args)...);
    for (INode *child : node->getChildren())
      func(child, std::forward<Args>(args)...);
    for (INode *child : node->getChildren())
      bfsTraverse(child, func, std::forward<Args>(args)...);
  }
}

#endif