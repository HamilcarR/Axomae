#ifndef ABSTRACTNODE_H
#define ABSTRACTNODE_H
#include "NodeInterface.h"

#include <internal/macro/project_macros.h>

namespace datastructure {
  class AbstractNode : public NodeInterface {

   protected:
    bool mark{false};                        /*<Generic mark , for graph traversal*/
    bool updated{false};                     /*<Lets the owning structure know if node has been modified */
    std::string name{};                      /*<Name of the node*/
    std::vector<NodeInterface *> parents{};  /*<List of parents*/
    std::vector<NodeInterface *> children{}; /*<List of children*/
    AbstractHierarchy *owner{};              /*<Structure owning this hierarchy*/
   public:
    ax_no_discard const std::vector<NodeInterface *> &getChildren() const override { return children; };
    ax_no_discard const std::vector<NodeInterface *> &getParents() const override { return parents; }
    ax_no_discard bool isLeaf() const override;
    ax_no_discard bool isMarked() const override { return mark; };
    ax_no_discard const std::string &getName() const override { return name; };
    ax_no_discard AbstractHierarchy *getHierarchyOwner() const override { return owner; };
    ax_no_discard bool isRoot() const override;
    ax_no_discard bool isUpdated() const override { return updated; }

    NodeInterface *returnRoot() override;
    void setMark(bool marked) override { mark = marked; };
    void setName(const std::string &str) override { name = str; };
    void setHierarchyOwner(AbstractHierarchy *_owner) override { owner = _owner; };
    void reset() override;
    void setParents(std::vector<NodeInterface *> &parents) override;
    void addChildNode(NodeInterface *node) override;
    void addParentNode(NodeInterface *node) override;
    void emptyParents() override;
    void emptyChildren() override;

    ~AbstractNode() override = default;
    AbstractNode(const AbstractNode &other) = default;
    AbstractNode(AbstractNode &&other) noexcept = default;
    AbstractNode &operator=(const AbstractNode &other) = default;
    AbstractNode &operator=(AbstractNode &&other) noexcept = default;

   protected:
    AbstractNode() = default;
  };
}  // namespace datastructure

#endif  // ABSTRACTNODE_H
