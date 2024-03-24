#ifndef MESHVISITOR_H
#define MESHVISITOR_H
#include "Node.h"
#include "Visitor.h"
class MeshVisitor final : public IVisitor<NodeInterface> {
 public:
  bool process(const NodeInterface *element) const override {}
};

#endif
