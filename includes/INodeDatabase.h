#ifndef INODEDATABASE_H
#define INODEDATABASE_H

#include "Axomae_macros.h"
#include "Node.h"
#include "RenderingDatabaseInterface.h"

class INodeDatabase : IResourceDB<int, INode> {
  using NodeIdMap = std::map<int, std::unique_ptr<INode>>;

 public:
  INodeDatabase();
  void clean() override;
  void purge() override;
  INode *get(const int id) override;
  bool remove(const int id) override;
  bool remove(const INode *element) override;
  int add(std::unique_ptr<INode> element, bool keep = false) override;
  bool contains(const int id) override;
  std::pair<int, INode *> contains(const INode *element_address) override;

 private:
  NodeIdMap database;
};

#endif