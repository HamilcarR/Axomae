#ifndef MESHVISITOR_H
#define MESHVISITOR_H
#include "Node.h"
#include "Visitor.h"
class MeshVisitor final : public IVisitor<INode> {
 public:
  bool process(const INode *element) const override {}
};

#endif
