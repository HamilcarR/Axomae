#ifndef INODEBUILDER_H
#define INODEBUILDER_H
#include "Axomae_macros.h"
#include "Logger.h"
#include "Node.h"
#include "constants.h"

/*Use for allocation tracking , and handling cyclic dependencies*/
class IBuilder {};

class NodeBuilder : public IBuilder {
 public:
  template<class NodeType, class... Args>
  static std::unique_ptr<NodeType> build(Args &&...args) {
    ASSERT_SUBTYPE(INode, NodeType);
    return std::make_unique<NodeType>(std::forward<Args>(args)...);
  }
};
#endif