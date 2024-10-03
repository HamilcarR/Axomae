#ifndef INODEFACTORY_H
#define INODEFACTORY_H

#include "Node.h"
#include "internal/common/Factory.h"
#include "internal/debug/Logger.h"
#include "internal/macro/project_macros.h"
/**
 * @file INodeFactory.h
 * class definition of a node factory
 *
 */

/**
 * @class NodeBuilder
 *
 */
class NodeBuilder {

 public:
  /**
   * @brief This method will only create an instance of node . The responsibility of taking care of the ownership of the returned instance relies on
   * the caller
   * @tparam NodeType Type of the node.
   * @tparam Args
   * @param args
   * @return std::unique_ptr<NodeType>
   */
  template<class NodeType, class... Args>
  static std::unique_ptr<NodeType> build(Args &&...args) {
    ASSERT_SUBTYPE(datastructure::NodeInterface, NodeType);
    return std::make_unique<PRVINTERFACE<NodeType, Args...>>(std::forward<Args>(args)...);
  }
};
#endif