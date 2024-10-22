#ifndef RENDERINGDATABASEINTERFACE_H
#define RENDERINGDATABASEINTERFACE_H

#include "DatabaseInterface.h"
#include "OperatorProgressStatus.h"
#include "constants.h"
#include "internal/memory/MemoryArena.h"
#include "internal/thread/Mutex.h"
#include <internal/common/exception/GenericException.h>
#include <internal/common/string/axomae_str_utils.h>
#include <internal/debug/Logger.h>
#include <map>
#include <unordered_map>

namespace core::memory {
  template<class T>
  class MemoryArena;
}

namespace exception {
  class DatabaseCacheAllocationException : public GenericException {
   public:
    DatabaseCacheAllocationException() : GenericException() { saveErrorString("Error allocating memory for database arena."); }
  };
}  // namespace exception
/**
 * @brief This class provides a non instanciable abstract class , to group common database methods.
 * @class IResourceDB
 * @tparam U Index of the stored object . Must be enum/numeric
 * @tparam T Class type of the object stored in the database
 */
template<class U, class T>
class IResourceDB : public DatabaseInterface<U, T>, public ILockable, public controller::IProgressManager {
 private:
  using CACHE = core::memory::MemoryArena<std::byte>;
  CACHE *memory_arena{nullptr};

  struct cache_block_t {
    uint8_t *cache_address{nullptr};
    std::size_t cache_size_bytes{0};
  };
  /* A map of all allocated arena memory for this instance of database , in bytes */
  using CACHELIST = std::unordered_map<uint8_t *, std::size_t>;
  CACHELIST reserved_memory_map;
  cache_block_t current_cache_block{nullptr, 0};
  std::size_t cache_increment{0};

 protected:
  using DATABASE = std::map<U, database::Storage<U, T>>;
  DATABASE database_map;

 public:
  void clean() override;
  void purge() override;

  bool remove(U id) override;
  bool remove(const T *element) override;
  ax_no_discard T *get(U id) const override;
  ax_no_discard U firstFreeId() const override = 0;
  ax_no_discard database::Result<U, T> add(std::unique_ptr<T> element, bool keep) override;
  ax_no_discard database::Result<U, T> contains(const T *element_address) const override;
  ax_no_discard int size() const override;
  bool contains(U id) const override;
  bool empty() const override;

  ax_no_discard const DATABASE &getConstData() const;
  bool setPersistence(U id, bool persistence = true);
  void setUpCacheMemory(CACHE *memory_cache);
  /**
   * Reserves an amount of contiguous memory in the arena
   */
  uint8_t *reserveCache(std::size_t size_bytes);
  /**
   * Deallocates caches from the arena for later use.
   */
  void invalidateCaches() override;
  ax_no_discard std::size_t getCacheSize(uint8_t *ptr) const override;
  ax_no_discard std::size_t getCurrentCacheSize() const override;
  ax_no_discard uint8_t *getCurrentCache() const override;
};

/* Some methods may have an ambiguous behavior depending on the type of the ID . this class provides a specialization of the
 * firstFreeId() method for integers id based databases*/
template<class T>
class IntegerResourceDB : public IResourceDB<int, T> {

 public:
  /**
   * @brief returns the first free ID of the map.
   * Returns either :\n
   * 1) first slot in which storage is marked invalid.\n
   * 2) first "Hole" between two valid slots. (ex : 1 -- [no slot] -- 3 , returns 2).\n
   * 3) allocated new storage in the map , at the end of the structure.\n
   * @return U id
   */
  ax_no_discard int firstFreeId() const override {
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
  invalidateCaches();
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

template<class U, class T>
uint8_t *IResourceDB<U, T>::reserveCache(std::size_t size_bytes) {
  if (!memory_arena)
    return nullptr;
  uint8_t *allocated = nullptr;
  Mutex::Lock lock(mutex);
  allocated = static_cast<uint8_t *>(memory_arena->allocate(size_bytes));
  if (!allocated) {
    LOG("Error allocating cache database.", LogLevel::CRITICAL);
    throw exception::DatabaseCacheAllocationException();
  }
  reserved_memory_map[allocated] = size_bytes;
  current_cache_block = {allocated, size_bytes};
  return allocated;
}

template<class U, class T>
std::size_t IResourceDB<U, T>::getCacheSize(uint8_t *ptr) const {
  try {
    return reserved_memory_map.at(ptr);
  } catch (const std::out_of_range &e) {
    LOG("Cannot retrieve cache size for cache address: " + utils::string::to_hex(reinterpret_cast<std::uintptr_t>(ptr)), LogLevel::WARNING);
    return 0;
  }
}

template<class U, class T>
std::size_t IResourceDB<U, T>::getCurrentCacheSize() const {
  return current_cache_block.cache_size_bytes;
}

template<class U, class T>
uint8_t *IResourceDB<U, T>::getCurrentCache() const {
  return current_cache_block.cache_address;
}

template<class U, class T>
void IResourceDB<U, T>::invalidateCaches() {
  if (!memory_arena) {
    LOG("Cache is not initialized.", LogLevel::ERROR);
    return;
  }
  Mutex::Lock lock(mutex);
  for (const auto &cache_buffer : reserved_memory_map)
    memory_arena->deallocate(cache_buffer.first);
  reserved_memory_map.clear();
}

template<class U, class T>
void IResourceDB<U, T>::setUpCacheMemory(CACHE *memory_cache) {
  memory_arena = memory_cache;
}
#endif