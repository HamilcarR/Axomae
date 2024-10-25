#ifndef DATABASESTORAGE_H
#define DATABASESTORAGE_H
#include <internal/macro/project_macros.h>
#include <memory>

namespace database {

  template<class T>
  struct storage_deleter_interface {
    virtual ~storage_deleter_interface() = default;
    virtual void operator()(T *ptr) const = 0;
  };

  template<class T>
  struct cached_storage_deleter : public storage_deleter_interface<T> {
    void operator()(T *ptr) const override { ptr->~T(); }
  };
  template<class T>
  struct default_storage_deleter : public storage_deleter_interface<T> {
    void operator()(T *ptr) const override { delete ptr; }
  };

  template<class T>
  struct VirtualDeleter {
    std::unique_ptr<storage_deleter_interface<T>> deleter;
    VirtualDeleter() = default;
    explicit VirtualDeleter(std::unique_ptr<storage_deleter_interface<T>> deleter_) : deleter(std::move(deleter_)) {}
    void operator()(T *ptr) const {
      if (deleter)
        (*deleter)(ptr);
    }
  };

  /**
   * @brief Stores objects inside the database
   * @tparam U ID type
   * @tparam T Obj type
   */
  template<class U, class T>
  class Storage {
    using storage_ptr = std::unique_ptr<T, VirtualDeleter<T>>;

   private:
    storage_ptr object{};
    U id;
    bool persistent{false};
    bool valid{false};

   public:
    Storage() = default;
    /*
     * Creates a storage with a unique pointer that owns completely the resource and is totally responsible for its lifecycle,
     * and calls delete on the resource at Storage end-life.
     */
    Storage(std::unique_ptr<T> object, U id, bool persistent);
    /*
     * Creates a storage with a unique pointer that owns the resource but doesn't destroy it.
     * This is useful for when the resource is present on an allocated buffer or a cache, that is destroyed by the caller.
     */
    Storage(T *object, U id, bool persistent);
    Storage(const Storage &) = delete;
    Storage &operator=(const Storage &) = delete;
    Storage(Storage &&assign) noexcept;
    Storage &operator=(Storage &&assign) noexcept;
    virtual ~Storage() = default;
    bool operator==(const Storage &compare);
    T *get() const;
    void setId(U id_);
    U getId() const;
    bool isPersistent() const;
    bool isValid() const;
    void setPersistence(bool pers);
    void setValidity(bool validity);
  };

  template<class U, class T>
  Storage<U, T>::Storage(std::unique_ptr<T> object_, U id_, bool persistent_) : id(id_), persistent(persistent_), valid(true) {
    std::unique_ptr<storage_deleter_interface<T>> deleter;
    deleter = std::make_unique<default_storage_deleter<T>>();
    object = storage_ptr(object_.release(), VirtualDeleter<T>(std::move(deleter)));
  }

  template<class U, class T>
  Storage<U, T>::Storage(T *object_, U id_, bool persistent_) : id(id_), persistent(persistent_), valid(true) {
    std::unique_ptr<storage_deleter_interface<T>> deleter;
    deleter = std::make_unique<cached_storage_deleter<T>>();
    object = storage_ptr(object_, VirtualDeleter<T>(std::move(deleter)));
  }

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
