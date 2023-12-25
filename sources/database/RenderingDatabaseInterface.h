#ifndef RENDERINGDATABASEINTERFACE_H
#define RENDERINGDATABASEINTERFACE_H
#include "Mutex.h"
#include "constants.h"
#include <map>
/**
 * @file RenderingDatabaseInterface.h
 * This file implements an interface for databases of objects like textures and meshes.
 * It's used as an owning structure over those resources.
 * When resources are needed , a pointer , or an ID is returned.
 * The structure takes care of managing a resource's lifespan.
 *
 */

namespace database {
  template<class IDTYPE, class OBJTYPE>
  struct Result {
    IDTYPE id;
    OBJTYPE *object;
  };

  // !Replace the whole negative id system bullshit with this :
  template<class U, class T>
  struct Storage {
    U id;
    std::unique_ptr<T> object;
    bool persistent;
  };

};  // namespace database

/**
 * @brief This class provides an interface of pure abstract methods to manage rendering objects databases
 * @class IResourceDB
 * @tparam T Class type of the object stored in the database
 */
template<class U, class T>
class IResourceDB {
  using DATABASE = std::map<U, database::Storage<U, T>>;

 public:
  /**
   * @brief Proceeds with a soft clean of the database . The implementation depends on the class that inherits this ,
   * but this usually consists of only some objects being freed
   *
   */
  virtual void clean() = 0;

  /**
   * @brief Proceeds with a complete purge of the database . Everything is freed .
   *
   */
  virtual void purge() {
    Mutex::Lock lock(mutex);
    database_map.clear();
  }

  /**
   * @brief Return a pointer on a single element , after a search using an ID .
   * @param id ID of the element
   * @return T* Pointer on the element
   */
  virtual T *get(const U id) const {
    Mutex::Lock lock(mutex);
    typename std::map<U, std::unique_ptr<T>>::const_iterator it = database_map.find(id);
    return it == database_map.end() ? nullptr : it->second.get();
  }

  /**
   * @brief Removes the element from a database , using it's ID
   *
   * @param id ID to remove
   * @return true If the element has been found and removed
   * @return false If no element with this ID have been found
   */
  virtual bool remove(const U id) {
    Mutex::Lock lock(mutex);
    typename std::map<U, std::unique_ptr<T>>::const_iterator it = database_map.find(id);
    if (it != database_map.end()) {
      database_map.erase(it);
      return true;
    }
    return false;
  }

  /**
   * @brief Removes an element from the database using it's address
   *
   * @param element Address of the element to be removed
   * @return true If the element has been removed
   * @return false If the element has not been found
   */
  virtual bool remove(const T *element) {
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

  /**
   * @brief Adds an element in the database
   *
   * @param element Object to store
   * @param keep Keep the element between scene change
   */
  virtual database::Result<U, T> add(std::unique_ptr<T> element, bool keep) = 0;

  /**
   * @brief Checks if database contains an object with specific ID .
   *
   * @param id ID of the element sought.
   */
  virtual bool contains(const U id) const {
    Mutex::Lock lock(mutex);
    return database_map.find(id) != database_map.end();
  }

  /**
   * @brief Checks if the element is present , if element not present , returns nullptr
   *
   * @param element_address address of the element we check
   * @return T* Pointer on the element
   */
  virtual database::Result<U, T> contains(const T *element_address) const {
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
  virtual bool empty() const { return database_map.empty(); }
  virtual int size() const { return database_map.size(); }
  virtual const std::map<U, std::unique_ptr<T>> &getConstData() const = 0;

 protected:
  mutable Mutex mutex;
  std::map<U, std::unique_ptr<T>> database_map;
};

#endif