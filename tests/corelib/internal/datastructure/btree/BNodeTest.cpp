#include "internal/datastructure/btree/BNode.hpp"
#include "Test.h"

using namespace datastructure::btree;

TEST(BNodeTest, isLeaf) {
  BNode<void *> leaf;
  EXPECT_TRUE(leaf.isLeaf());
  BNode<void *> one_child;
  one_child.setLeft(&leaf);
  EXPECT_FALSE(one_child.isLeaf());
}

TEST(BNodeTest, isRoot) {
  BNode<void *> root;
  EXPECT_TRUE(root.isRoot());
  BNode<void *> parent;
  root.setParent(&parent);
  EXPECT_FALSE(root.isRoot());
}

TEST(BNodeTest, setChildren) {
  BNode<void *> root;
  BNode<void *> child;
  root.setLeft(&child);
  EXPECT_NE(root.left(), nullptr);
  EXPECT_EQ(root.right(), nullptr);
  BNode<void *> child2;
  root.setRight(&child2);
  EXPECT_NE(root.right(), nullptr);
  EXPECT_NE(root.left(), nullptr);
}
