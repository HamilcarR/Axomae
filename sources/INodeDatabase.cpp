#include "../includes/INodeDatabase.h"
#include <algorithm>
INodeDatabase::INodeDatabase() {}

void INodeDatabase::clean() {
  Mutex::Lock lock(mutex);
  for (const auto &A : database) {
    if (A.first > 0)
      database.erase(A.first);
  }
}

void INodeDatabase::purge() {
  Mutex::Lock lock(mutex);
  database.clear();
}

INode *INodeDatabase::get(const int id) const {
  Mutex::Lock lock(mutex);
  NodeIdMap::const_iterator it = database.find(id);
  return it == database.end() ? nullptr : it->second.get();
}

bool INodeDatabase::remove(const int id) {
  Mutex::Lock lock(mutex);
  if (database[id] != nullptr) {
    database.erase(id);
    return true;
  }
  return false;
}
bool INodeDatabase::remove(const INode *element) {
  Mutex::Lock lock(mutex);
  for (auto &A : database) {
    if (element == A.second.get()) {
      database.erase(A.first);
      return true;
    }
  }
  return false;
}

/* See TextureDatabase::add() , same process */
int INodeDatabase::add(std::unique_ptr<INode> element, bool keep) {
  Mutex::Lock lock(mutex);
  if (keep) {
    int index = -1;
    while (database[index] != nullptr)
      index--;
    database[index] = std::move(element);
    return index;
  } else {
    int index = 0;
    while (database[index] != nullptr)
      index++;
    database[index] = std::move(element);
    return index;
  }
}
bool INodeDatabase::contains(const int id) const {
  Mutex::Lock lock(mutex);
  return database.find(id) != database.end();
}
std::pair<int, INode *> INodeDatabase::contains(const INode *element_address) const {
  Mutex::Lock lock(mutex);
  for (const auto &A : database) {
    if (A.second.get() == element_address)
      return std::pair(A.first, A.second.get());
  }
  return std::pair(INFINITY, nullptr);
}
