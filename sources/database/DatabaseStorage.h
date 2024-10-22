#ifndef DATABASESTORAGE_H
#define DATABASESTORAGE_H
#include <internal/macro/project_macros.h>
#include <memory>

namespace database {
  /**
   * @brief Stores objects inside the database
   * @tparam U ID type
   * @tparam T Obj type
   */
  template<class U, class T>
  class Storage {
   private:
    std::unique_ptr<T> object;
    U id;
    bool persistent;
    bool valid;

   public:
    Storage();
    Storage(std::unique_ptr<T> object_, U id_, bool persistent_);
    Storage(const Storage &) = delete;
    Storage &operator=(const Storage &) = delete;
    Storage(Storage &&assign) noexcept;
    Storage &operator=(Storage &&assign) noexcept;
    virtual ~Storage() = default;
    bool operator==(const Storage &compare);
    T *get() const;
    void setId(U id_);
    U getId() const;
    ax_no_discard bool isPersistent() const;
    ax_no_discard bool isValid() const;
    void setPersistence(bool pers);
    void setValidity(bool validity);
  };

  template<class U, class T>
  Storage<U, T>::Storage() : object(nullptr), persistent(false), valid(false) {}

  template<class U, class T>
  Storage<U, T>::Storage(std::unique_ptr<T> object_, U id_, bool persistent_)
      : object(std::move(object_)), id(id_), persistent(persistent_), valid(true) {}

  template<class U, class T>
  Storage<U, T>::Storage(Storage &&assign) noexcept {
    id = assign.id;
    object = std::move(assign.object);
    persistent = assign.persistent;
    valid = assign.valid;
    assign.valid = false;
  }

  template<class U, class T>
  Storage<U, T> &Storage<U, T>::operator=(Storage<U, T> &&assign) noexcept {
    id = assign.id;
    object = std::move(assign.object);
    persistent = assign.persistent;
    valid = assign.valid;
    assign.valid = false;
    return *this;
  }

  template<class U, class T>
  bool Storage<U, T>::operator==(const Storage<U, T> &compare) {
    bool id_comp = id == compare.id;
    bool obj_comp = object.get() == compare.object.get();
    bool persist_comp = persistent == compare.persistent;
    bool validity = valid == compare.valid;
    return id_comp && obj_comp && persist_comp && validity;
  }

  template<class U, class T>
  T *Storage<U, T>::get() const {
    return object.get();
  }

  template<class U, class T>
  void Storage<U, T>::setId(U id_) {
    id = id_;
  }

  template<class U, class T>
  U Storage<U, T>::getId() const {
    return id;
  }

  template<class U, class T>
  ax_no_discard bool Storage<U, T>::isPersistent() const {
    return persistent;
  }

  template<class U, class T>
  ax_no_discard bool Storage<U, T>::isValid() const {
    return valid;
  }

  template<class U, class T>
  void Storage<U, T>::setPersistence(bool pers) {
    persistent = pers;
  }

  template<class U, class T>
  void Storage<U, T>::setValidity(bool validity) {
    valid = validity;
  }

}  // namespace database
#endif  // DATABASESTORAGE_H
