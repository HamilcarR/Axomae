#ifndef DatabaseInterface_H
#define DatabaseInterface_H
#include "DatabaseStorage.h"
#include "QueryResult.h"

/**
 * @file DatabaseInterface.h
 * This file implements an interface for databases of objects like textures and meshes.
 * It's used as an owning structure over those resources.
 * When resources are needed , a pointer , or an ID is returned.
 * The structure takes care of managing a resource's lifespan.
 *
 */

/**
 * @class DatabaseInterface
 * @tparam U ID type
 * @tparam T Object type
 */
template<class U, class T>
class DatabaseInterface {
  /**
   * @brief Proceeds with a soft clean of the database . The implementation depends on the class that inherits this ,
   * but this usually consists of only some objects being freed , according to their persistence.
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
   * @return T* Pointer on the element or nullptr if not found
   */
  virtual T *get(U id) const = 0;

  /**
   * @brief Removes the element from a database , using it's ID
   *
   * @param id ID to remove
   * @return true If the element has been found and removed
   * @return false If no element with this ID have been found
   */
  virtual bool remove(U id) = 0;

  /**
   * @brief Removes an element from the database using it's address
   *
   * @param element Address of the element to be removed
   * @return true If the element has been removed
   * @return false If the element has not been found
   */
  virtual bool remove(const T *element) = 0;

  [[nodiscard]] virtual U firstFreeId() const = 0;

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
  virtual bool contains(U id) const = 0;

  /**
   * @brief Checks if the element is present , if element not present , returns a Result with object = nullptr
   *
   * @param element_address address of the element we check
   * @return database::Result
   */
  virtual database::Result<U, T> contains(const T *element_address) const = 0;
  [[nodiscard]] virtual bool empty() const = 0;
  [[nodiscard]] virtual int size() const = 0;
};
#endif  // DatabaseInterface_H
