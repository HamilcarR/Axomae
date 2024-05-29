#ifndef BNODE_H
#define BNODE_H
#include "AbstractNode.h"
#include "Axomae_macros.h"
#include "BNodeInterface.h"
namespace datastructure::btree {
  template<class T>
  class BNode : public AbstractNode, public BNodeInterface<T> {

   protected:
    T node_data;

   public:
    BNode();
    explicit BNode(T &data);
    ~BNode() override = default;
    BNode(const BNode &other) = default;
    BNode(BNode &&other) noexcept = default;
    BNode &operator=(const BNode &other) = default;
    BNode &operator=(BNode &&other) noexcept = default;
    void setLeft(BNode *node) override;
    void setRight(BNode *node) override;
    void setParent(BNode *node) override;
    [[nodiscard]] BNode *right() const override;
    [[nodiscard]] BNode *left() const override;
    const T &data() const override;
    void setData(T &data) override;

   private:
    void setParents(std::vector<NodeInterface *> &parents) override;
    void addChildNode(NodeInterface *node) override;
    void addParentNode(NodeInterface *node) override;
  };

  template<class T>
  BNode<T>::BNode() {
    parents.resize(1);
    children.resize(2);
  }

  template<class T>
  BNode<T>::BNode(T &d) : BNode() {
    node_data(d);
  }

  template<class T>
  BNode<T> *BNode<T>::right() const {
    return static_cast<BNode *>(getChildren()[1]);
  }

  template<class T>
  BNode<T> *BNode<T>::left() const {
    return static_cast<BNode *>(getChildren()[0]);
  }

  template<class T>
  const T &BNode<T>::data() const {
    return node_data;
  }

  template<class T>
  void BNode<T>::setData(T &data) {
    node_data = data;
  }

  template<class T>
  void BNode<T>::setRight(BNode *node) {
    children[1] = node;
  }
  template<class T>
  void BNode<T>::setLeft(BNode *node) {
    children[0] = node;
  }
  template<class T>
  void BNode<T>::setParent(BNode *node) {
    parents[0] = node;
  }
  template<class T>
  void BNode<T>::setParents(std::vector<NodeInterface *> &parents_) {
    EMPTY_FUNCBODY
  }
  template<class T>
  void BNode<T>::addChildNode(NodeInterface *node) {
    EMPTY_FUNCBODY
  }

  template<class T>
  void BNode<T>::addParentNode(NodeInterface *node) {
    EMPTY_FUNCBODY
  }

}  // namespace datastructure::btree
#endif  // BNODE_H
