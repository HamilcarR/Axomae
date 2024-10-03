#include "internal/datastructure/btree/BTree.hpp"
#include "Test.h"

using namespace datastructure::btree;

TEST(BTreeTest, setRootNode) {
  BNode<void *> node;
  BNode<void *> node1;
  node.setLeft(&node1);
  BTree<void *> tree(&node);
  EXPECT_EQ(tree.getRootNode(), &node);
}

TEST(BTreeTest, findName) {
  BNode<void *> child1;
  child1.setName("child1");
  BNode<void *> child2;
  child1.setLeft(&child2);
  BNode<void *> root;
  root.setRight(&child1);
  BTree<void *> tree(&root);
  auto collection = tree.findByName("child1");
  ASSERT_EQ(collection.size(), 1);
  EXPECT_EQ(collection[0], &child1);
}