#ifndef RENDERINGDATABASEINTERFACE_H
#define RENDERINGDATABASEINTERFACE_H
#include "Mutex.h"
#include "constants.h"
#include <map>
/**
 * @file RenderingDatabaseInterface.h
 * This file implements an interface for databases of objects , like , for example , light databases.
 *
 */

namespace database {
  template<class IDTYPE, class OBJTYPE>
  struct Result {
    IDTYPE id;
    OBJTYPE *object;
  };

};  // namespace database

/**
 * @brief This class provides an interface of pure abstract methods to manage rendering objects databases
 * @class IResourceDB
 * @tparam T Class type of the object stored in the database
 */
template<class U, class T>
class IResourceDB {
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
  virtual void purge() = 0;

  /**
   * @brief Return a pointer on a single element , after a search using an ID .
   * @param id ID of the element
   * @return T* Pointer on the element
   */
  virtual T *get(const U id) const = 0;

  /**
   * @brief Removes the element from a database , using it's ID
   *
   * @param id ID to remove
   * @return true If the element has been found and removed
   * @return false If no element with this ID have been found
   */
  virtual bool remove(const U id) = 0;

  /**
   * @brief Removes an element from the database using it's address
   *
   * @param element Address of the element to be removed
   * @return true If the element has been removed
   * @return false If the element has not been found
   */
  virtual bool remove(const T *element) = 0;

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
  virtual bool contains(const U id) const = 0;

  /**
   * @brief Checks if the element is present , if element not present , returns nullptr
   *
   * @param element_address address of the element we check
   * @return T* Pointer on the element
   */
  virtual database::Result<U, T> contains(const T *element_address) const = 0;

  virtual const std::map<U, std::unique_ptr<T>> &getConstData() const = 0;

 protected:
  mutable Mutex mutex;
  std::map<U, std::unique_ptr<T>> database;
};

#endif