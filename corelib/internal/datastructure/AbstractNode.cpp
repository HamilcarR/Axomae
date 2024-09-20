#include "AbstractNode.h"

#include <algorithm>
using namespace datastructure;

bool AbstractNode::isLeaf() const {
  bool children_null = true;
  for (NodeInterface *A : children)
    if (A != nullptr) {
      children_null = false;
      break;
    }
  return children.empty() || children_null;
}

bool AbstractNode::isRoot() const {
  if (parents.empty())
    return true;
  for (NodeInterface *A : parents)
    if (A != nullptr)
      return false;
  return true;
}

void AbstractNode::emptyParents() { parents.clear(); }

void AbstractNode::emptyChildren() { children.clear(); }

void AbstractNode::reset() {
  mark = false;
  updated = false;
  name = "";
  emptyChildren();
  emptyParents();
  owner = nullptr;
}

/**
 * TODO : To be changed in case graphs are implemented for a more correct graph traversal algo
 */
NodeInterface *AbstractNode::returnRoot() {
  parents.erase(std::remove(parents.begin(), parents.end(), nullptr), parents.end());
  NodeInterface *iterator = getParents()[0];
  if (iterator == nullptr)
    return this;
  while (iterator->getParents()[0] != nullptr)
    iterator = iterator->getParents()[0];
  return iterator;
}

void AbstractNode::setParents(std::vector<NodeInterface *> &nodes) {
  if (!nodes.empty()) {
    parents.clear();
    parents.push_back(nodes[0]);
    if (parents[0] != nullptr)
      parents[0]->addChildNode(this);
  }
}

void AbstractNode::addParentNode(NodeInterface *node) {
  if (node != nullptr) {
    parents.push_back(node);
    node->addChildNode(this);
  }
}
void AbstractNode::addChildNode(NodeInterface *node) {
  if (node) {
    bool contains = std::find(children.begin(), children.end(), node) != children.end();
    if (!contains) {
      children.push_back(node);
      std::vector<NodeInterface *> ret = {this};
      node->setParents(ret);
    }
  }
}
