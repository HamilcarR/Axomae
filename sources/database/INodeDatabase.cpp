#include "INodeDatabase.h"
#include <algorithm>

INodeDatabase::INodeDatabase() {}

void INodeDatabase::clean() {
  Mutex::Lock lock(mutex);
  std::vector<int> to_delete;
  for (auto &A : database_map) {
    if (A.first >= 0) {
      A.second->clean();
      to_delete.push_back(A.first);
    }
  }
  for (const auto A : to_delete) {
    database_map.erase(A);
  }
}

void INodeDatabase::purge() {
  Mutex::Lock lock(mutex);
  for (auto &A : database_map)
    A.second->clean();
  database_map.clear();
}

INode *INodeDatabase::get(const int id) const {
  Mutex::Lock lock(mutex);
  NodeIdMap::const_iterator it = database_map.find(id);
  return it == database_map.end() ? nullptr : it->second.get();
}

bool INodeDatabase::remove(const int id) {
  Mutex::Lock lock(mutex);
  auto it = database_map.find(id);
  if (it != database_map.end()) {
    it->second->clean();
    database_map.erase(it);
    return true;
  }
  return false;
}
bool INodeDatabase::remove(const INode *element) {
  Mutex::Lock lock(mutex);
  for (auto &A : database_map) {
    if (element == A.second.get()) {
      A.second->clean();
      database_map.erase(A.first);
      return true;
    }
  }
  return false;
}

/* See TextureDatabase::add() , same process */
database::Result<int, INode> INodeDatabase::add(std::unique_ptr<INode> element, bool keep) {
  Mutex::Lock lock(mutex);
  if (keep) {
    int index = -1;
    while (database_map[index] != nullptr)
      index--;
    database_map[index] = std::move(element);
    return {index, database_map[index].get()};
  } else {
    int index = 0;
    while (database_map[index] != nullptr)
      index++;
    database_map[index] = std::move(element);
    return {index, database_map[index].get()};
  }
}
bool INodeDatabase::contains(const int id) const {
  Mutex::Lock lock(mutex);
  return database_map.find(id) != database_map.end();
}
database::Result<int, INode> INodeDatabase::contains(const INode *element_address) const {
  Mutex::Lock lock(mutex);
  for (const auto &A : database_map) {
    if (A.second.get() == element_address)
      return {A.first, A.second.get()};
  }
  return {0, nullptr};
}
