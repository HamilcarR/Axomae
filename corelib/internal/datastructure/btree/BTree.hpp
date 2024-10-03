#ifndef BTREE_H
#define BTREE_H
#include "BNode.hpp"
#include "internal/datastructure/AbstractHierarchy.h"
namespace datastructure::btree {
  template<class T>
  class BTree : public AbstractHierarchy {

   public:
    BTree() = default;
    explicit BTree(BNode<T> *root_);
    ~BTree() override = default;
    BTree(const BTree &other) = default;
    BTree(BTree &&other) noexcept = default;
    BTree &operator=(const BTree &other) = default;
    BTree &operator=(BTree &&other) noexcept = default;
  };

  template<class T>
  BTree<T>::BTree(BNode<T> *root_) {
    root = root_;
  }

}  // namespace datastructure::btree
#endif  // BTREE_H
