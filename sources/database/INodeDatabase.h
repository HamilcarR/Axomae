#ifndef INODEDATABASE_H
#define INODEDATABASE_H

#include "Axomae_macros.h"
#include "Factory.h"
#include "Node.h"
#include "RenderingDatabaseInterface.h"

class INodeDatabase final : public IntegerResourceDB<NodeInterface> {
  using NodeIdMap [[maybe_unused]] = std::map<int, std::unique_ptr<NodeInterface>>;

 public:
  explicit INodeDatabase(controller::ProgressStatus *progress_manager = nullptr);
  ~INodeDatabase() override = default;
  void clean() override;
  void purge() override;
  NodeInterface *get(int id) const override;
  bool remove(int id) override;
  bool remove(const NodeInterface *element) override;
  bool contains(int id) const override;
  database::Result<int, NodeInterface> contains(const NodeInterface *element_address) const override;
};

namespace database::node {
  /* Builds an NodeInterface and store into specified database */
  template<class NodeType, class... Args>
  static database::Result<int, NodeType> store(IResourceDB<int, NodeInterface> &database, bool keep, Args &&...args) {
    ASSERT_SUBTYPE(NodeInterface, NodeType);
    std::unique_ptr<NodeType> node = std::make_unique<PRVINTERFACE<NodeType, Args...>>(std::forward<Args>(args)...);
    database::Result<int, NodeInterface> result = database.add(std::move(node), keep);
    database::Result<int, NodeType> cast = {result.id, static_cast<NodeType *>(result.object)};
    return cast;
  }
};  // namespace database::node

#endif