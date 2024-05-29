#ifndef SCENEHIERARCHY_H
#define SCENEHIERARCHY_H
#include "AbstractHierarchy.h"
#include "HierarchyInterface.h"
#include "Node.h"
/**
 * @file SceneGraph.h
 * @brief This file implements the hierarchy system of a scene.
 */

/**
 * @class SceneTree
 */
class SceneTree : public datastructure::AbstractHierarchy {
 private:
  bool node_updated{};

 public:
  explicit SceneTree(SceneTreeNode *root = nullptr);
  ~SceneTree() override = default;
  SceneTree(const SceneTree &copy) = default;
  SceneTree(SceneTree &&move) noexcept = default;
  SceneTree &operator=(const SceneTree &copy) = default;
  SceneTree &operator=(SceneTree &&move) noexcept = default;
  /**
   * @brief Traverse the scene structure upwards, and recomputes all accum transformations
   */
  virtual void updateAccumulatedTransformations();
  virtual void pushNewRoot(SceneTreeNode *new_root);
};

#endif