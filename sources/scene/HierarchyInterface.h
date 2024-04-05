#ifndef HierarchyInterface_H
#define HierarchyInterface_H
#include "NodeInterface.h"
#include <queue>
#include <string>
#include <vector>

/**
 *@file HierarchyInterface.h
 */

template<class T, class U>
class IResourceDB;

class HierarchyInterface {
 public:
  virtual ~HierarchyInterface() = default;
  virtual void setRoot(NodeInterface *_root) = 0;
  [[nodiscard]] virtual NodeInterface *getIterator() const = 0;
  virtual void setIterator(NodeInterface *iter) = 0;
  virtual void createGenericRootNode(IResourceDB<int, NodeInterface> &database) = 0;
  [[nodiscard]] virtual NodeInterface *getRootNode() const = 0;
  virtual void setAsRootChild(NodeInterface *node) = 0;
  /**
   * @brief Traverse the scene structure, and set their owner of each node to this structure
   */
  virtual void updateOwner() = 0;
  /**
   * @brief Free only generic nodes  .
   * Other types of nodes need to be deallocated by their own structures and owners. For example , lights should be
   * freed by their LightDatabase , even though they are stored in the scene tree as well .
   */
  virtual void clear() = 0;
  virtual std::vector<NodeInterface *> findByName(const std::string &name) = 0;
};

/**
 * @class AbstractHierarchy
 */
class AbstractHierarchy : public HierarchyInterface {
 protected:
  NodeInterface *root{}; /*<Root of the hierarchy*/
 private:
  mutable NodeInterface *iterator{}; /*<Iterator to keep track of nodes in traversals*/
  mutable const NodeInterface *const_iterator{};

 protected:
  AbstractHierarchy() = default;

 public:
  ~AbstractHierarchy() override = default;
  AbstractHierarchy(const AbstractHierarchy &copy) = default;
  AbstractHierarchy(AbstractHierarchy &&move) = default;
  AbstractHierarchy &operator=(const AbstractHierarchy &copy) = default;
  AbstractHierarchy &operator=(AbstractHierarchy &&move) = default;
  void setRoot(NodeInterface *_root) override { root = _root; }
  NodeInterface *getIterator() const override { return iterator; }
  void createGenericRootNode(IResourceDB<int, NodeInterface> &database) override;
  void setIterator(NodeInterface *iter) override { iterator = iter; }
  NodeInterface *getRootNode() const override { return root; }
  void setAsRootChild(NodeInterface *node) override;
  void updateOwner() override;
  void clear() override;

  template<class F, class... Args>
  void dfs(NodeInterface *begin, F func, Args &&...args);

  template<class F, class... Args>
  void dfsConst(const NodeInterface *begin, F func, Args &&...args) const;

  template<class F, class... Args>
  void bfs(NodeInterface *begin, F func, Args &&...args);

  template<class F, class... Args>
  void bfsConst(const NodeInterface *begin, F func, Args &&...args) const;

 private:
  template<class F, class... Args>
  void dfsTraverse(NodeInterface *node, F func, Args &&...args);

  template<class F, class... Args>
  void dfsConstTraverse(const NodeInterface *node, F func, Args &&...args) const;

  template<class F, class... Args>
  void bfsTraverse(NodeInterface *node, F func, Args &&...args);

  template<class F, class... Args>
  void bfsConstTraverse(const NodeInterface *node, F func, Args &&...args) const;
};

template<class F, class... Args>
void AbstractHierarchy::dfs(NodeInterface *begin, F func, Args &&...args) {
  if ((iterator = begin) != nullptr) {
    dfsTraverse(iterator, func, std::forward<Args>(args)...);
  }
}

template<class F, class... Args>
void AbstractHierarchy::dfsTraverse(NodeInterface *node, F func, Args &&...args) {
  if (node != nullptr) {
    func(node, std::forward<Args>(args)...);
    for (NodeInterface *child : node->getChildren())
      dfsTraverse(child, func, std::forward<Args>(args)...);
  }
}

template<class F, class... Args>
void AbstractHierarchy::dfsConst(const NodeInterface *begin, F func, Args &&...args) const {
  if ((const_iterator = begin) != nullptr) {
    dfsTraverse(const_iterator, func, std::forward<Args>(args)...);
  }
}

template<class F, class... Args>
void AbstractHierarchy::dfsConstTraverse(const NodeInterface *node, F func, Args &&...args) const {
  if (node != nullptr) {
    func(node, std::forward<Args>(args)...);
    for (const NodeInterface *child : node->getChildren())
      dfsTraverse(child, func, std::forward<Args>(args)...);
  }
}

template<class F, class... Args>
void AbstractHierarchy::bfs(NodeInterface *begin, F func, Args &&...args) {
  if ((iterator = begin) != nullptr) {
    bfsTraverse(iterator, func, std::forward<Args>(args)...);
  }
}
#include <iostream>
template<class F, class... Args>
void AbstractHierarchy::bfsTraverse(NodeInterface *node, F func, Args &&...args) {
  if (!node)
    return;
  std::queue<NodeInterface *> queue;
  queue.push(node);
  while (!queue.empty()) {
    NodeInterface *node_processed = queue.front();
    queue.pop();
    func(node_processed, std::forward<Args>(args)...);
    for (NodeInterface *child : node_processed->getChildren())
      queue.push(child);
  }
}

template<class F, class... Args>
void AbstractHierarchy::bfsConst(const NodeInterface *begin, F func, Args &&...args) const {
  if ((const_iterator = begin) != nullptr) {
    bfsConstTraverse(const_iterator, func, std::forward<Args>(args)...);
  }
}

template<class F, class... Args>
void AbstractHierarchy::bfsConstTraverse(const NodeInterface *node, F func, Args &&...args) const {
  if (!node)
    return;
  std::queue<const NodeInterface *> queue;
  queue.push(node);
  while (!queue.empty()) {
    const NodeInterface *node_processed = queue.front();
    queue.pop();
    func(node_processed, std::forward<Args>(args)...);
    for (const NodeInterface *child : node_processed->getChildren())
      queue.push(child);
  }
}

#endif  // HierarchyInterface_H
