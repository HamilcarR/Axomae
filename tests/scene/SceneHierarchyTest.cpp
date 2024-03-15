#include "SceneHierarchy.h"
#include "INodeDatabase.h"
#include "INodeFactory.h"
#include "Test.h"

constexpr unsigned int ITERATIONS = 5;
constexpr unsigned int TEST_TREE_MAX_DEPTH = 4;
constexpr unsigned int TEST_TREE_MAX_NODE_DEGREE = 3;
class SceneTreeBuilder {
 public:
  SceneTreeBuilder() {
    srand(time(nullptr));
    node_count = 0;
    leaf_count = 0;
  }

  virtual ~SceneTreeBuilder() = default;

  void buildSceneTree(unsigned int depth, unsigned int max_degree) {
    ISceneNode *root = database::node::store<SceneTreeNode>(database, false).object;
    root->setName("root");
    tree.setRoot(root);
    node_count++;
    buildRecursive(root, max_degree, depth);
    tree.updateOwner();
  }

  SceneTree *getTreePointer() { return &tree; }

  unsigned getLeafCount() { return leaf_count; }

  unsigned getNodeCount() { return node_count; }

  void clean() { tree.clear(); }

  std::vector<INode *> findByName(const std::string &name) {
    std::vector<INode *> names;
    findByNameRecursive(tree.getRootNode(), name, names);
    return names;
  }

 private:
  void findByNameRecursive(INode *node, const std::string &name, std::vector<INode *> &collection) {
    if (node != nullptr) {
      if (node->getName() == name)
        collection.push_back(node);
      for (auto A : node->getChildren())
        findByNameRecursive(A, name, collection);
    }
    return;
  }

  void buildRecursive(ISceneNode *node, unsigned max_degree, unsigned depth) {
    if (depth == 0) {
      leaf_count++;
      return;
    }
    int size_numbers = rand() % max_degree + 1;
    for (int i = 0; i < size_numbers; i++) {
      ISceneNode *child = database::node::store<SceneTreeNode>(database, false, node).object;
      child->setName(std::string("child-") + std::to_string(depth) + std::string("-") + std::to_string(i));
      node_count++;
      buildRecursive(child, max_degree, depth - 1);
    }
  }

 protected:
  SceneTree tree;
  unsigned node_count;
  unsigned leaf_count;
  INodeDatabase database;
};

class PseudoFunctors {
 public:
  static void testDummyFunction(INode *node) { std::cout << node << "\n"; }
  static void testNodeCount(INode *node, unsigned int *i) { (*i)++; }
  static void testTransformationPropagation(INode *node, std::vector<glm::mat4> &matrices) {
    glm::mat4 m = static_cast<ISceneNode *>(node)->computeFinalTransformation();
    matrices.push_back(m);
  }
  static void testLeafCount(INode *node, unsigned int *leaf_number) {
    if (node->isLeaf())
      (*leaf_number)++;
  }
};

TEST(DFSTest, dfsNodeCount) {
  SceneTreeBuilder builder;
  builder.buildSceneTree(TEST_TREE_MAX_DEPTH, TEST_TREE_MAX_NODE_DEGREE);
  SceneTree *tree = builder.getTreePointer();
  unsigned int i = 0;
  tree->dfs(tree->getRootNode(), &PseudoFunctors::testNodeCount, &i);
  builder.clean();
  EXPECT_EQ(i, builder.getNodeCount());
}

TEST(DFSTest, updateAccumulatedTransformations) {
  SceneTreeBuilder builder;
  builder.buildSceneTree(TEST_TREE_MAX_DEPTH, TEST_TREE_MAX_NODE_DEGREE);
  SceneTree *tree = builder.getTreePointer();
  auto root_local_transf = static_cast<ISceneNode *>(tree->getRootNode())->getLocalModelMatrix();
  auto updated_local_transf = glm::translate(root_local_transf, glm::vec3(1., 0., 0.));
  static_cast<ISceneNode *>(tree->getRootNode())->setLocalModelMatrix(updated_local_transf);
  tree->updateAccumulatedTransformations();
  std::vector<glm::mat4> matrices;
  tree->dfs(tree->getRootNode(), &PseudoFunctors::testTransformationPropagation, matrices);
  for (auto m : matrices) {
    bool eq = m == updated_local_transf;
    ASSERT_EQ(eq, true);
  }
  builder.clean();
}

TEST(DFSTest, leafCount) {
  SceneTreeBuilder builder;
  builder.buildSceneTree(TEST_TREE_MAX_DEPTH, TEST_TREE_MAX_NODE_DEGREE);
  SceneTree *tree = builder.getTreePointer();
  unsigned i = 0;
  tree->dfs(tree->getRootNode(), &PseudoFunctors::testLeafCount, &i);
  EXPECT_EQ(i, builder.getLeafCount());
}

TEST(DFSTest, findName) {
  SceneTreeBuilder builder;
  builder.buildSceneTree(TEST_TREE_MAX_DEPTH, TEST_TREE_MAX_NODE_DEGREE);
  SceneTree *tree = builder.getTreePointer();
  std::vector<std::string> names;
  names.push_back("root");
  for (unsigned i = 0; i < ITERATIONS; i++) {
    std::string test_string = std::string("child-") + std::to_string(rand() % TEST_TREE_MAX_DEPTH + 1) + std::string("-") +
                              std::to_string(rand() % TEST_TREE_MAX_NODE_DEGREE + 1);
    names.push_back(test_string);
  }
  for (auto A : names) {
    auto test_result = builder.findByName(A);
    auto dfs_result = tree->findByName(A);
    EXPECT_EQ(test_result, dfs_result);
  }
}

TEST(BFSTEST, leafCountBFS) {
  SceneTreeBuilder builder;
  builder.buildSceneTree(TEST_TREE_MAX_DEPTH, TEST_TREE_MAX_NODE_DEGREE);
  SceneTree *tree = builder.getTreePointer();
  unsigned i = 0;
  tree->bfs(tree->getRootNode(), &PseudoFunctors::testLeafCount, &i);
  EXPECT_EQ(i, builder.getLeafCount());
}
