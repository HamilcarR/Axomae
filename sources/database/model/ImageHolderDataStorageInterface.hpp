#ifndef IMAGEHOLDERDATASTORAGEINTERFACE
#define IMAGEHOLDERDATASTORAGEINTERFACE

#include <cstddef>
template<class T>
class ImageHolderDataStorageInterface {
 public:
  virtual ~ImageHolderDataStorageInterface() = default;

  virtual const T *data() const = 0;

  virtual T *data() = 0;

  virtual std::size_t size() const = 0;

  virtual const T &operator[](std::size_t index) const = 0;

  virtual T &operator[](std::size_t index) = 0;

  virtual void reserve(std::size_t size) = 0;

  virtual void resize(std::size_t size, const T &value = T()) = 0;

  virtual T *begin() { return data(); };

  virtual T *end() { return data() + size(); };

  virtual const T *begin() const { return data(); };

  virtual const T *end() const { return data() + size(); };

  virtual void clear() = 0;

  virtual bool empty() const { return size() == 0; }
};

#endif
