#ifndef RENDERINGDATABASEINTERFACE_H
#define RENDERINGDATABASEINTERFACE_H

#include "DatabaseInterface.h"
#include "OperatorProgressStatus.h"
#include "constants.h"
#include "internal/memory/MemoryArena.h"
#include "internal/thread/Mutex.h"
#include <internal/common/exception/GenericException.h>
#include <internal/common/string/string_utils.h>
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

constexpr std::size_t DATABASE_CACHE_INVALID_SIZE = -1;

/**
 * @brief This class provides a non instanciable abstract class , to group common database methods.
 * @class IResourceDB
 * @tparam U Index of the stored object . Must be enum/numeric
 * @tparam T Class type of the object stored in the database
 */
template<class U, class T>
class IResourceDB : public DatabaseInterface<U, T>, public ILockable, public controller::IProgressManager {
 private:
  using CACHE = core::memory::ByteArena;
  CACHE *memory_arena{nullptr};

  struct cache_block_t {
    uint8_t *cache_address{nullptr};
    std::size_t cache_element_position{0};
    std::size_t cache_size_bytes{0};
    std::string cache_label{""};
  };
  /* A map of all allocated arena memory for this instance of database , in bytes */
  using CACHELIST = std::unordered_map<uint8_t *, cache_block_t>;
  CACHELIST reserved_memory_map;
  cache_block_t *current_working_cache{nullptr};

 protected:
  using DATABASE = std::map<U, database::Storage<U, T>>;
  DATABASE database_map;

 private:
  ax_no_discard const cache_block_t *getCacheBlock(uint8_t *ptr) const;
  ax_no_discard cache_block_t *getCacheBlock(uint8_t *ptr);
  ax_no_discard cache_block_t *getCurrentCacheBlock() const;

 public:
  void clean() override;
  void purge() override;

  bool remove(U id) override;
  bool remove(const T *element) override;
  ax_no_discard T *get(U id) const override;
  ax_no_discard U firstFreeId() const override = 0;
  database::Result<U, T> add(std::unique_ptr<T> element, bool keep) override;
  /**
   * Uses placement to create elements in the arena through the cache_address.
   * It is up to the caller to allocate first enough memory.
   */
  template<class SUBTYPE, class... Args>
  database::Result<U, T> addCachedElement(bool keep, uint8_t *cache_address, Args &&...args);
  /**
   * Copy array or buffer `to_copy` inside the specified cache, and returns it's in-buffer address.
   * if `cache_address` is null , will copy inside the current cache.
   */
  template<class TYPE>
  TYPE *copyRangeToCache(TYPE *to_copy, uint8_t *cache_address, std::size_t count, std::size_t offset);
  ax_no_discard database::Result<U, T> contains(const T *element_address) const override;
  ax_no_discard int size() const override;
  bool contains(U id) const override;
  bool empty() const override;

  ax_no_discard const DATABASE &getConstData() const;
  bool setPersistence(U id, bool persistence = true);
  void setUpCacheMemory(CACHE *memory_cache);
  /**
   * Reserves an amount of contiguous memory in the arena,
   * and sets the current cache address with the newly created address.
   */
  uint8_t *reserveCache(std::size_t size_bytes, std::size_t alignment, const char *label = nullptr);
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
  U ffid = firstFreeId();
  T *ptr = element.get();
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
uint8_t *IResourceDB<U, T>::reserveCache(std::size_t size_bytes, std::size_t alignment, const char *label) {
  if (!memory_arena)
    return nullptr;
  uint8_t *allocated = nullptr;
  Mutex::Lock lock(mutex);
  allocated = static_cast<uint8_t *>(memory_arena->allocate(size_bytes, alignment));
  if (!allocated) {
    LOG("Error allocating cache database.", LogLevel::CRITICAL);
    return nullptr;
  }
  cache_block_t block;
  block.cache_size_bytes = size_bytes;
  block.cache_address = allocated;
  block.cache_element_position = 0;
  if (label)
    block.cache_label = label;
  else
    block.cache_label = "";
  reserved_memory_map[allocated] = block;
  current_working_cache = &reserved_memory_map.at(allocated);
  return allocated;
}

template<class U, class T>
std::size_t IResourceDB<U, T>::getCacheSize(uint8_t *ptr) const {
  try {
    return reserved_memory_map.at(ptr).cache_size_bytes;
  } catch (const std::out_of_range &e) {
    LOG("Cannot retrieve cache size for cache address: " + utils::string::to_hex(reinterpret_cast<std::uintptr_t>(ptr)), LogLevel::WARNING);
    return DATABASE_CACHE_INVALID_SIZE;
  }
}

template<class U, class T>
std::size_t IResourceDB<U, T>::getCurrentCacheSize() const {
  if (!current_working_cache)
    return DATABASE_CACHE_INVALID_SIZE;
  return current_working_cache->cache_size_bytes;
}

template<class U, class T>
uint8_t *IResourceDB<U, T>::getCurrentCache() const {
  if (!current_working_cache)
    return nullptr;
  return current_working_cache->cache_address;
}

template<class U, class T>
typename IResourceDB<U, T>::cache_block_t *IResourceDB<U, T>::getCurrentCacheBlock() const {
  return current_working_cache;
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

template<class U, class T>
const typename IResourceDB<U, T>::cache_block_t *IResourceDB<U, T>::getCacheBlock(uint8_t *ptr) const {
  if (getCacheSize(ptr) == DATABASE_CACHE_INVALID_SIZE)
    return nullptr;
  return &reserved_memory_map.at(ptr);
}

template<class U, class T>
typename IResourceDB<U, T>::cache_block_t *IResourceDB<U, T>::getCacheBlock(uint8_t *ptr) {
  if (getCacheSize(ptr) == DATABASE_CACHE_INVALID_SIZE)
    return nullptr;
  return &reserved_memory_map.at(ptr);
}

template<class U, class T>
template<class SUBTYPE, class... Args>
database::Result<U, T> IResourceDB<U, T>::addCachedElement(bool keep, uint8_t *cache_address, Args &&...args) {
  ASSERT_SUBTYPE(SUBTYPE, T);
  if (!memory_arena) {
    LOG("Cache is not initialized.", LogLevel::ERROR);
    return database::Result<U, T>();
  }
  cache_block_t *cache_block = getCacheBlock(cache_address);
  if (!cache_block)
    return database::Result<U, T>();
  std::size_t offset = cache_block->cache_element_position * sizeof(SUBTYPE);
  if (offset >= cache_block->cache_size_bytes) {
    LOG("Block offset reached max block size.", LogLevel::ERROR);
    return database::Result<U, T>();
  }
  Mutex::Lock lock(mutex);
  SUBTYPE *created_resource = memory_arena->constructAtMemPosition<SUBTYPE>(
      reinterpret_cast<SUBTYPE *>(cache_address), cache_block->cache_element_position, std::forward<Args>(args)...);
  if (!created_resource) {
    LOG("Error creating resource at cache address: " + utils::string::to_hex(reinterpret_cast<uintptr_t>(cache_address + offset)), LogLevel::ERROR);
    return database::Result<U, T>();
  }
  cache_block->cache_element_position += 1;
  U ffid = firstFreeId();
  T *ptr = created_resource;
  /* Since we're allocating on a buffer we own , we need a custom deleter that doesn't call delete on arena memory . */
  database::Storage<U, T> storage(created_resource, ffid, keep);
  database_map[ffid] = std::move(storage);
  return {ffid, ptr};
}

template<class U, class T>
template<class TYPE>
TYPE *IResourceDB<U, T>::copyRangeToCache(TYPE *to_copy, uint8_t *cache_address, std::size_t count, std::size_t offset) {
  if (!cache_address)
    cache_address = getCurrentCache();
  if (!getCacheBlock(cache_address)) {
    LOG("Cache address :" + utils::string::to_hex(reinterpret_cast<uintptr_t>(cache_address)) + " not initialized", LogLevel::ERROR);
    return nullptr;
  }
  if (!memory_arena) {
    LOG("Cache is not initialized", LogLevel::ERROR);
    return nullptr;
  }
  if (!to_copy) {
    LOG("Invalid buffer address", LogLevel::ERROR);
    return nullptr;
  }
  return static_cast<TYPE *>(memory_arena->copyRange(to_copy, cache_address, count * sizeof(TYPE), offset * sizeof(TYPE)));
}

#endif