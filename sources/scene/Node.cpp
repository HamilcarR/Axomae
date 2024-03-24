#include "Node.h"
#include <algorithm>

SceneTreeNode::SceneTreeNode(SceneTreeNode *_parent, AbstractHierarchy *_owner) {
  if (_parent != nullptr) {
    std::vector<NodeInterface *> ret = {_parent};
    setParents(ret);
  }
  setHierarchyOwner(_owner);
  local_transformation = accumulated_transformation = glm::mat4(1.f);
  mark = false;
  name = "Generic-Hierarchy-Node";
  updated = true;
}

SceneTreeNode::SceneTreeNode(const std::string &_name, const glm::mat4 &transformation, SceneTreeNode *parent, AbstractHierarchy *owner)
    : SceneTreeNode(parent, owner) {
  name = _name;
  local_transformation = transformation;
}

bool SceneTreeNode::isLeaf() const { return children.empty(); }

bool SceneTreeNode::isRoot() const {
  if (parents.empty())
    return true;
  else {
    for (NodeInterface *A : parents)
      if (A != nullptr)
        return false;
    return true;
  }
}

void SceneTreeNode::emptyParents() { parents.clear(); }

void SceneTreeNode::emptyChildren() { children.clear(); }

void SceneTreeNode::reset() {
  mark = false;
  updated = false;
  name = "";
  emptyChildren();
  emptyParents();
  owner = nullptr;
  resetLocalModelMatrix();
  resetAccumulatedMatrix();
}

void SceneTreeNode::resetLocalModelMatrix() { local_transformation = glm::mat4(1.f); }

void SceneTreeNode::resetAccumulatedMatrix() { accumulated_transformation = glm::mat4(1.f); }

/**
 * The function returns the root node of a scene tree by traversing up the parent nodes until the root
 * is reached.
 * @return a pointer to a SceneNodeInterface object, specifically the root node of the scene tree.
 */
NodeInterface *SceneTreeNode::returnRoot() {
  parents.erase(std::remove(parents.begin(), parents.end(), nullptr), parents.end());
  SceneTreeNode *iterator = getParent();
  if (iterator == nullptr)
    return this;
  while (iterator->getParent() != nullptr)
    iterator = iterator->getParent();
  return iterator;
}

/**
 * The function computes the final transformation matrix for a scene tree node by multiplying the
 * accumulated transformation matrix with the local transformation matrix.
 * @return a glm::mat4, which is a 4x4 matrix representing a transformation.
 */
glm::mat4 SceneTreeNode::computeFinalTransformation() {
  if (parents.empty() || parents[0] == nullptr)
    return local_transformation;
  else
    return accumulated_transformation * local_transformation;
}

void SceneTreeNode::setParents(std::vector<NodeInterface *> &nodes) {
  if (!nodes.empty()) {
    parents.clear();
    parents.push_back(nodes[0]);
    if (parents[0] != nullptr)
      parents[0]->addChildNode(this);
  }
}

void SceneTreeNode::setParent(NodeInterface *node) {
  if (node != nullptr) {
    std::vector<NodeInterface *> ret = {node};
    setParents(ret);
  }
}

SceneTreeNode *SceneTreeNode::getParent() const {
  if (!parents.empty())
    return dynamic_cast<SceneTreeNode *>(parents[0]);
  else
    return nullptr;
}

void SceneTreeNode::addChildNode(NodeInterface *node) {
  if (node) {
    bool contains = std::find(children.begin(), children.end(), node) != children.end();
    if (!contains) {
      children.push_back(node);
      std::vector<NodeInterface *> ret = {this};
      node->setParents(ret);
    }
  }
}
