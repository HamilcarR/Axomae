#ifndef INODEDATABASE_H
#define INODEDATABASE_H

#include "Axomae_macros.h"
#include "Factory.h"
#include "Node.h"
#include "RenderingDatabaseInterface.h"

class INodeDatabase : public IResourceDB<int, INode> {
  using NodeIdMap = std::map<int, std::unique_ptr<INode>>;

 public:
  INodeDatabase();
  void clean() override;
  void purge() override;
  INode *get(const int id) const override;
  bool remove(const int id) override;
  bool remove(const INode *element) override;
  database::Result<int, INode> add(std::unique_ptr<INode> element, bool keep = false) override;
  bool contains(const int id) const override;
  database::Result<int, INode> contains(const INode *element_address) const override;
  const NodeIdMap &getConstData() const override { return database_map; }

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