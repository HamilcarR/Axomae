#include "INodeDatabase.h"
#include <algorithm>
using namespace datastructure;
INodeDatabase::INodeDatabase(controller::ProgressStatus *progress_manager_) { progress_manager = progress_manager_; }

void INodeDatabase::clean() {
  Mutex::Lock lock(mutex);
  std::vector<int> to_delete;
  for (auto &A : database_map) {
    if (!A.second.isPersistent()) {
      A.second.get()->reset();
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
    A.second.get()->reset();
  database_map.clear();
}

NodeInterface *INodeDatabase::get(const int id) const {
  Mutex::Lock lock(mutex);
  auto it = database_map.find(id);
  return it == database_map.end() ? nullptr : it->second.get();
}

bool INodeDatabase::remove(const int id) {
  Mutex::Lock lock(mutex);
  auto it = database_map.find(id);
  if (it != database_map.end()) {
    it->second.get()->reset();
    database_map.erase(it);
    return true;
  }
  return false;
}
bool INodeDatabase::remove(const NodeInterface *element) {
  Mutex::Lock lock(mutex);
  for (auto &A : database_map) {
    if (element == A.second.get()) {
      A.second.get()->reset();
      database_map.erase(A.first);
      return true;
    }
  }
  return false;
}

bool INodeDatabase::contains(const int id) const { return database_map.find(id) != database_map.end(); }

database::Result<int, NodeInterface> INodeDatabase::contains(const NodeInterface *element_address) const {
  Mutex::Lock lock(mutex);
  for (const auto &A : database_map) {
    if (A.second.get() == element_address)
      return {A.first, A.second.get()};
  }
  return {-1, nullptr};
}
