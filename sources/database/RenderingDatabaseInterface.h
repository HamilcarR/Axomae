#ifndef RENDERINGDATABASEINTERFACE_H
#define RENDERINGDATABASEINTERFACE_H
#include "Mutex.h"
#include "OP_ProgressStatus.h"
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
  class Result {
   public:
    IDTYPE id;
    OBJTYPE *object;
    Result() : id{}, object(nullptr) {}
    Result(IDTYPE id_, OBJTYPE *object_) : id(id_), object(object_) {}
    Result(const Result &copy) : id(copy.id), object(copy.object) {}
    Result(Result &&assign) noexcept : id(std::move(assign.id)), object(assign.object) {}
    Result &operator=(Result &&assign) noexcept {
      id = std::move(assign.id);
      object = assign.object;
      return *this;
    }
    Result &operator=(const Result &copy) {
      if (this != &copy) {
        id = copy.id;
        object = copy.object;
      }
      return *this;
    }
    virtual ~Result() = default;

    bool operator!=(const Result<IDTYPE, OBJTYPE> &compare) const {
      bool eq = *this == compare;
      return !eq;
    }
    bool operator==(const Result<IDTYPE, OBJTYPE> &compare) const { return id == compare.id && object == compare.object; }
  };

  template<class U, class T>
  class Storage {
   public:
    Storage() {
      valid = false;
      persistent = false;
      object = nullptr;
    }
    ~Storage() = default;
    Storage(std::unique_ptr<T> object_, U id_, bool persistent_) {
      object = std::move(object_);
      id = id_;
      persistent = persistent_;
      valid = true;
    }
    Storage(const Storage &) = delete;
    Storage &operator=(const Storage &) = delete;
    Storage(Storage &&assign) noexcept {
      id = assign.id;
      object = std::move(assign.object);
      persistent = assign.persistent;
      valid = assign.valid;
      assign.valid = false;
    }

    Storage &operator=(Storage &&assign) noexcept {
      id = assign.id;
      object = std::move(assign.object);
      persistent = assign.persistent;
      valid = assign.valid;
      assign.valid = false;
      return *this;
    }

    bool operator==(const Storage &compare) {
      bool id_comp = id == compare.id;
      bool obj_comp = object.get() == compare.object.get();
      bool persist_comp = persistent == compare.persistent;
      bool validity = valid == compare.valid;
      return id_comp && obj_comp && persist_comp && validity;
    }

    T *get() const { return object.get(); }
    void setId(U id_) { id = id_; }
    void getId() const { return id; }

    [[nodiscard]] bool isPersistent() const { return persistent; }
    [[nodiscard]] bool isValid() const { return valid; }
    void setPersistence(bool pers) { persistent = pers; }
    void setValidity(bool validity) { valid = validity; }

   private:
    std::unique_ptr<T> object;
    U id;
    bool persistent;
    bool valid;
  };

};  // namespace database

/**
 * @brief This class provides an interface of pure abstract methods to manage rendering objects databases
 * @class IResourceDB
 * @tparam U Index of the stored object . Must be enum/numeric
 * @tparam T Class type of the object stored in the database
 */
template<class U, class T>
class IResourceDB {
 protected:
  using DATABASE = std::map<U, database::Storage<U, T>>;

 public:
  /**
   * @brief Proceeds with a soft clean of the database . The implementation depends on the class that inherits this ,
   * but this usually consists of only some objects being freed , according to their persistence.
   *
   */
  virtual void clean() {
    std::vector<typename DATABASE::const_iterator> delete_list;
    for (typename DATABASE::const_iterator it = database_map.begin(); it != database_map.end(); it++) {
      if (!it->second.isPersistent())
        delete_list.push_back(it);
    }
    Mutex::Lock lock(mutex);
    for (auto &elem : delete_list)
      database_map.erase(elem);
  }

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
   * @return T* Pointer on the element or nullptr if not found
   */
  virtual T *get(const U id) const {
    Mutex::Lock lock(mutex);
    typename DATABASE::const_iterator it = database_map.find(id);
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
    typename DATABASE::const_iterator it = database_map.find(id);
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

  [[nodiscard]] virtual U firstFreeId() const = 0;

  /**
   * @brief Adds an element in the database
   *
   * @param element Object to store
   * @param keep Keep the element between scene change
   */
  virtual database::Result<U, T> add(std::unique_ptr<T> element, bool keep) {
    T *ptr = element.get();
    U ffid = firstFreeId();
    database::Storage<U, T> storage(std::move(element), ffid, keep);
    Mutex::Lock lock(mutex);
    database_map[ffid] = std::move(storage);
    return {ffid, ptr};
  }

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
   * @brief Checks if the element is present , if element not present , returns a Result with object = nullptr
   *
   * @param element_address address of the element we check
   * @return database::Result
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
  const DATABASE &getConstData() const { return database_map; }
  bool setPersistence(const U id, bool persistence = true) {
    auto it = database_map.find(id);
    if (it == database_map.end())
      return false;
    it->second.setPersistence(persistence);
    return true;
  }

  [[nodiscard]] controller::ProgressStatus *getProgressManager() const { return progress_manager; }
  void setProgressManager(controller::ProgressStatus *progress_m) { progress_manager = progress_m; }

 protected:
  mutable Mutex mutex;
  DATABASE database_map;
  controller::ProgressStatus *progress_manager;
};

/* Some methods may have an ambiguous behavior depending on the type of the ID . this class provides a specialization of the
 * firstFreeId() method for integers id based databases*/
template<class T>
class IntegerResourceDB : public IResourceDB<int, T> {
  using BASETYPE = IResourceDB<int, T>;

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
  [[nodiscard]] virtual int firstFreeId() const {
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

#endif