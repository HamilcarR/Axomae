#include "SceneHierarchy.h"
#include "INodeDatabase.h"
#include "INodeFactory.h"

using namespace datastructure;

SceneTree::SceneTree(SceneTreeNode *node) { root = node; }

void SceneTree::updateAccumulatedTransformations() {
  if (root != nullptr) {
    auto recompute_matrices = [](NodeInterface *n) {
      auto *node = dynamic_cast<SceneTreeNode *>(n);
      if (!node->isRoot()) {
        auto *parent = dynamic_cast<SceneTreeNode *>(node->getParents()[0]);
        glm::mat4 new_accum = parent->computeFinalTransformation();
        node->setAccumulatedModelMatrix(new_accum);
      } else
        node->setAccumulatedModelMatrix(glm::mat4(1.f));
    };
    dfs(root, recompute_matrices);
  }
}

void SceneTree::pushNewRoot(SceneTreeNode *new_root) {
  if (root != new_root) {
    if (root == nullptr)
      root = new_root;
    else {
      NodeInterface *temp = root;
      new_root->emptyChildren();
      new_root->addChildNode(root);
      new_root->emptyParents();
      root = new_root;
      std::vector<NodeInterface *> new_parent = {new_root};
      temp->setParents(new_parent);
    }
    updateOwner();
  }
}
