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
    ASSERT_SUBTYPE(INode, NodeType);
    return std::make_unique<PRVINTERFACE<NodeType, Args...>>(std::forward<Args>(args)...);
  }

  /**
   * @brief This method will automatically build and move a node to a specified IResourceDB
   * @tparam NodeType Type of the node
   * @tparam Args
   * @param args
   * @param database Database we want to store our nodes in.
   * @param keep Keep nodes between scene change.
   * @return int ID of the node in the database .
   */
  template<class NodeType, class... Args>
  static database::Result<int, NodeType> store(IResourceDB<int, INode> &database, bool keep, Args &&...args) {
    ASSERT_SUBTYPE(INode, NodeType);
    std::unique_ptr<NodeType> node = std::make_unique<PRVINTERFACE<NodeType, Args...>>(std::forward<Args>(args)...);
    database::Result<int, INode> result = database.add(std::move(node), keep);
    database::Result<int, NodeType> cast = {result.id, static_cast<NodeType *>(result.object)};
    return cast;
  }
};
#endif