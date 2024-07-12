#include "AbstractHierarchy.h"
using namespace datastructure;
void AbstractHierarchy::setAsRootChild(NodeInterface *node) {
  if (root != nullptr)
    root->addChildNode(node);
}

void AbstractHierarchy::updateOwner() {
  if (root != nullptr) {
    auto scene_update_owner = [](NodeInterface *node, AbstractHierarchy *owner) { node->setHierarchyOwner(owner); };
    dfs(root, scene_update_owner, this);
  }
}

void AbstractHierarchy::clear() { root = nullptr; }

std::vector<NodeInterface *> AbstractHierarchy::findByName(const std::string &name) {
  auto lambda_search_name = [](NodeInterface *node, const std::string &name, std::vector<NodeInterface *> &collection) {
    if (node->getName() == name)
      collection.push_back(node);
  };
  std::vector<NodeInterface *> collection;
  dfs(root, lambda_search_name, name, collection);
  return collection;
}
