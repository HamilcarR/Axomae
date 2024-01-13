#ifndef INODEDATABASE_H
#define INODEDATABASE_H

#include "Axomae_macros.h"
#include "Factory.h"
#include "Node.h"
#include "RenderingDatabaseInterface.h"

class INodeDatabase final : public IntegerResourceDB<INode> {
  using NodeIdMap = std::map<int, std::unique_ptr<INode>>;

 public:
  INodeDatabase() = default;
  void clean() override;
  void purge() override;
  INode *get(int id) const override;
  bool remove(int id) override;
  bool remove(const INode *element) override;
  bool contains(int id) const override;
  database::Result<int, INode> contains(const INode *element_address) const override;

 private:
};

namespace database::node {
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
};  // namespace database::node

#endif