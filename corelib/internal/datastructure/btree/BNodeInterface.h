#ifndef BNODEINTERFACE_H
#define BNODEINTERFACE_H
#include <internal/macro/project_macros.h>

namespace datastructure::btree {
  template<class T>
  class BNode;

  template<class T>
  class BNodeInterface {
   public:
    virtual ~BNodeInterface() = default;
    virtual const T &data() const = 0;
    virtual void setData(T &data) = 0;
    virtual void setLeft(BNode<T> *node) = 0;
    virtual void setRight(BNode<T> *node) = 0;
    virtual void setParent(BNode<T> *node) = 0;
    ax_no_discard virtual BNode<T> *right() const = 0;
    ax_no_discard virtual BNode<T> *left() const = 0;
  };
}  // namespace datastructure::btree

#endif  // BNODEINTERFACE_H
