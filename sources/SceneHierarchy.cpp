#include "../includes/SceneHierarchy.h"

void ISceneHierarchy::setAsRootChild(INode *node) {
  if (root != nullptr)
    root->addChildNode(node);
}

void ISceneHierarchy::updateOwner() {
  if (root != nullptr) {
    auto scene_update_owner = [](INode *node, ISceneHierarchy *owner) { node->setHierarchyOwner(owner); };
    dfs(root, scene_update_owner, this);
  }
}

void ISceneHierarchy::clear() {
  for (auto A : generic_nodes_to_delete) {
    delete A;
  }
  generic_nodes_to_delete.clear();
}

/*******************************************************************************************************************************************************************/

SceneTree::SceneTree(ISceneNode *node) {
  root = node;
}

SceneTree::SceneTree(const SceneTree &copy) {
  root = copy.getRootNode();
  setIterator(copy.getIterator());
}

SceneTree::~SceneTree() {}

void SceneTree::createGenericRootNode() {
  root = new SceneTreeNode();
}

SceneTree &SceneTree::operator=(const SceneTree &copy) {
  if (this != &copy) {
    root = copy.getRootNode();
    setIterator(copy.getIterator());
  }
  return *this;
}

void SceneTree::updateAccumulatedTransformations() {
  if (root != nullptr) {
    auto recompute_matrices = [](INode *n) {
      ISceneNode *node = static_cast<ISceneNode *>(n);
      if (!node->isRoot()) {
        glm::mat4 new_accum = static_cast<ISceneNode *>(node->getParents()[0])->computeFinalTransformation();
        node->setAccumulatedModelMatrix(new_accum);
      } else
        node->setAccumulatedModelMatrix(glm::mat4(1.f));
    };
    dfs(root, recompute_matrices);
  }
}

void SceneTree::pushNewRoot(INode *new_root) {
  if (root != new_root) {
    if (root == nullptr)
      root = new_root;
    else {
      INode *temp = root;
      new_root->emptyChildren();
      new_root->addChildNode(root);
      new_root->emptyParents();
      root = new_root;
      std::vector<INode *> new_parent = {new_root};
      temp->setParents(new_parent);
    }
    updateOwner();
  }
}

std::vector<INode *> SceneTree::findByName(const std::string &name) {
  auto lambda_search_name = [](INode *node, const std::string &name, std::vector<INode *> &collection) {
    if (node->getName() == name)
      collection.push_back(node);
  };
  std::vector<INode *> collection;
  dfs(root, lambda_search_name, name, collection);
  return collection;
}

void SceneTree::updatedHierarchy() {
  emit modifiedStructureEvent();
}