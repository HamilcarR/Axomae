#ifndef BTREE_H
#define BTREE_H
#include "HierarchyInterface.h"

namespace datastructure::btree {
  class BTree : public AbstractHierarchy {

   public:
    ~BTree() override = default;
  };

  class LinearBTree {};

}  // namespace datastructure::btree
#endif  // BTREE_H
