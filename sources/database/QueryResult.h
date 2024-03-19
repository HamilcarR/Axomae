#ifndef QUERYRESULT_H
#define QUERYRESULT_H
#include <algorithm>

namespace database {
  /**
   * @brief A database query result structure
   */
  template<class IDTYPE, class OBJTYPE>
  class Result {
   public:
    IDTYPE id;
    OBJTYPE *object;

   public:
    Result();
    Result(IDTYPE id_, OBJTYPE *object_);
    Result(const Result &copy);
    Result(Result &&assign) noexcept;
    Result &operator=(Result &&assign) noexcept;
    Result &operator=(const Result &copy);
    bool operator!=(const Result<IDTYPE, OBJTYPE> &compare) const;
    bool operator==(const Result<IDTYPE, OBJTYPE> &compare) const;
  };

  template<class T, class Y>
  Result<T, Y>::Result() : id{}, object(nullptr) {}

  template<class T, class Y>
  Result<T, Y>::Result(T id_, Y *object_) : id(id_), object(object_) {}

  template<class T, class Y>
  Result<T, Y>::Result(const Result &copy) : id(copy.id), object(copy.object) {}

  template<class T, class Y>
  Result<T, Y>::Result(Result &&assign) noexcept : id(std::move(assign.id)), object(assign.object) {}

  template<class T, class Y>
  Result<T, Y> &Result<T, Y>::operator=(Result<T, Y> &&assign) noexcept {
    id = std::move(assign.id);
    object = assign.object;
    return *this;
  }

  template<class T, class Y>
  Result<T, Y> &Result<T, Y>::operator=(const Result<T, Y> &copy) {
    if (this != &copy) {
      id = copy.id;
      object = copy.object;
    }
    return *this;
  }

  template<class T, class Y>
  bool Result<T, Y>::operator!=(const Result<T, Y> &compare) const {
    bool eq = *this == compare;
    return !eq;
  }

  template<class T, class Y>
  bool Result<T, Y>::operator==(const Result<T, Y> &compare) const {
    return id == compare.id && object == compare.object;
  }

}  // namespace database

#endif  // QUERYRESULT_H
