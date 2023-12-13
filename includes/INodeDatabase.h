#ifndef INODEDATABASE_H
#define INODEDATABASE_H

#include "Axomae_macros.h"
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

 private:
  NodeIdMap database;
};

#endif