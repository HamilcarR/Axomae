#ifndef RENDERINGDATABASEINTERFACE_H
#define RENDERINGDATABASEINTERFACE_H

#include "DatabaseInterface.h"
#include "Mutex.h"
#include "OP_ProgressStatus.h"
#include "constants.h"
#include <map>

/**
 * @brief This class provides a (pseudo) interface of pure abstract methods to manage rendering objects databases .
 * @class IResourceDB
 * @tparam U Index of the stored object . Must be enum/numeric
 * @tparam T Class type of the object stored in the database
 */
template<class U, class T>
class IResourceDB : public DatabaseInterface<U, T>, public ILockable, public controller::IProgressManager {
 protected:
  using DATABASE = std::map<U, database::Storage<U, T>>;

 protected:
  DATABASE database_map;

 public:
  void clean() override;
  void purge() override;
  T *get(U id) const override;
  bool remove(U id) override;
  bool remove(const T *element) override;
  [[nodiscard]] U firstFreeId() const override = 0;
  database::Result<U, T> add(std::unique_ptr<T> element, bool keep) override;
  bool contains(U id) const override;
  database::Result<U, T> contains(const T *element_address) const override;
  bool empty() const override;
  int size() const override;
  const DATABASE &getConstData() const;
  bool setPersistence(U id, bool persistence = true);
};

/* Some methods may have an ambiguous behavior depending on the type of the ID . this class provides a specialization of the
 * firstFreeId() method for integers id based databases*/
template<class T>
class IntegerResourceDB : public IResourceDB<int, T> {

 public:
  /**
   * @brief returns the first free ID of the map.
   *
   * Returns either :
   * 1) first slot in which storage is marked invalid .
   * 2) first "Hole" between two valid slots. (ex : 1 -- [no slot] -- 3 , returns 2)
   * 3) allocated new storage in the map , at the end.
   * @return U id
   */
  [[nodiscard]] int firstFreeId() const override {
    using BASETYPE = IResourceDB<int, T>;
    int diff = 0;
    if (BASETYPE::database_map.begin()->first > 0)  // In case 0 is available.
      return 0;
    for (const auto &elem : BASETYPE::database_map) {
      if (!elem.second.isValid())
        return elem.first;
      if ((elem.first - diff) > 1)
        return (elem.first + diff) / 2;
      diff = elem.first;
    }
    return BASETYPE::size();
  }
};

template<class U, class T>
void IResourceDB<U, T>::clean() {
  std::vector<typename DATABASE::const_iterator> delete_list;
  for (typename DATABASE::const_iterator it = database_map.begin(); it != database_map.end(); it++) {
    if (!it->second.isPersistent())
      delete_list.push_back(it);
  }
  Mutex::Lock lock(mutex);
  for (auto &elem : delete_list)
    database_map.erase(elem);
}

template<class U, class T>
void IResourceDB<U, T>::purge() {
  Mutex::Lock lock(mutex);
  database_map.clear();
}

template<class U, class T>
T *IResourceDB<U, T>::get(const U id) const {
  Mutex::Lock lock(mutex);
  typename DATABASE::const_iterator it = database_map.find(id);
  return it == database_map.end() ? nullptr : it->second.get();
}

template<class U, class T>
bool IResourceDB<U, T>::remove(const U id) {
  Mutex::Lock lock(mutex);
  typename DATABASE::const_iterator it = database_map.find(id);
  if (it != database_map.end()) {
    database_map.erase(it);
    return true;
  }
  return false;
}

template<class U, class T>
bool IResourceDB<U, T>::remove(const T *element) {
  Mutex::Lock lock(mutex);
  if (!element)
    return false;
  for (auto &A : database_map) {
    if (element == A.second.get()) {
      database_map.erase(A.first);
      return true;
    }
  }
  return false;
}

template<class U, class T>
database::Result<U, T> IResourceDB<U, T>::add(std::unique_ptr<T> element, bool keep) {
  T *ptr = element.get();
  U ffid = firstFreeId();
  database::Storage<U, T> storage(std::move(element), ffid, keep);
  Mutex::Lock lock(mutex);
  database_map[ffid] = std::move(storage);
  return {ffid, ptr};
}

template<class U, class T>
bool IResourceDB<U, T>::contains(const U id) const {
  Mutex::Lock lock(mutex);
  return database_map.find(id) != database_map.end();
}

template<class U, class T>
database::Result<U, T> IResourceDB<U, T>::contains(const T *element_address) const {
  Mutex::Lock lock(mutex);
  if (!element_address)
    return {static_cast<U>(0), nullptr};
  for (const auto &A : database_map) {
    if (A.second.get() == element_address) {
      return {A.first, A.second.get()};
    }
  }
  return {static_cast<U>(0), nullptr};
}

template<class U, class T>
bool IResourceDB<U, T>::empty() const {
  return database_map.empty();
}

template<class U, class T>
int IResourceDB<U, T>::size() const {
  return database_map.size();
}

template<class U, class T>
const typename IResourceDB<U, T>::DATABASE &IResourceDB<U, T>::getConstData() const {
  return database_map;
}

template<class U, class T>
bool IResourceDB<U, T>::setPersistence(const U id, bool persistence) {
  auto it = database_map.find(id);
  if (it == database_map.end())
    return false;
  it->second.setPersistence(persistence);
  return true;
}

#endif