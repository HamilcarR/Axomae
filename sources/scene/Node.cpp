#include "Node.h"
#include <algorithm>

SceneTreeNode::SceneTreeNode(SceneTreeNode *_parent, datastructure::AbstractHierarchy *_owner) {
  if (_parent != nullptr) {
    std::vector<NodeInterface *> ret = {_parent};
    SceneTreeNode::setParents(ret);
  }
  SceneTreeNode::setHierarchyOwner(_owner);
  local_transformation = accumulated_transformation = glm::mat4(1.f);
  mark = false;
  name = "Generic-Hierarchy-Node";
  updated = true;
}

SceneTreeNode::SceneTreeNode(const std::string &_name,
                             const glm::mat4 &transformation,
                             SceneTreeNode *parent,
                             datastructure::AbstractHierarchy *owner)
    : SceneTreeNode(parent, owner) {
  name = _name;
  local_transformation = transformation;
}

void SceneTreeNode::reset() {
  AbstractNode::reset();
  resetLocalModelMatrix();
  resetAccumulatedMatrix();
}

void SceneTreeNode::resetLocalModelMatrix() { local_transformation = glm::mat4(1.f); }

void SceneTreeNode::resetAccumulatedMatrix() { accumulated_transformation = glm::mat4(1.f); }

glm::mat4 SceneTreeNode::computeFinalTransformation() {
  if (ignore)
    return accumulated_transformation;
  else {
    if (parents.empty() || parents[0] == nullptr)
      return local_transformation;
    else
      return accumulated_transformation * local_transformation;
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
