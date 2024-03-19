#ifndef INODEDATABASE_H
#define INODEDATABASE_H

#include "Axomae_macros.h"
#include "Factory.h"
#include "Node.h"
#include "RenderingDatabaseInterface.h"

class INodeDatabase final : public IntegerResourceDB<INode> {
  using NodeIdMap [[maybe_unused]] = std::map<int, std::unique_ptr<INode>>;

 public:
  explicit INodeDatabase(controller::ProgressStatus *progress_manager = nullptr);
  void clean() override;
  void purge() override;
  INode *get(int id) const override;
  bool remove(int id) override;
  bool remove(const INode *element) override;
  bool contains(int id) const override;
  database::Result<int, INode> contains(const INode *element_address) const override;
};

namespace database::node {
  /* Builds an INode and store into specified database */
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