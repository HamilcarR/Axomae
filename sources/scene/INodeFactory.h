#ifndef INODEFACTORY_H
#define INODEFACTORY_H
#include "Axomae_macros.h"
#include "Factory.h"
#include "Logger.h"
#include "Node.h"
#include "RenderingDatabaseInterface.h"
#include "constants.h"

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
    ASSERT_SUBTYPE(NodeInterface, NodeType);
    return std::make_unique<PRVINTERFACE<NodeType, Args...>>(std::forward<Args>(args)...);
  }
};
#endif