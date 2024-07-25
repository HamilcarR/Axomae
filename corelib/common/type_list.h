#ifndef TYPE_LIST_H
#define TYPE_LIST_H
#include "project_macros.h"

namespace core {
  template<class... Ts>
  struct type_list {
    static constexpr int size = sizeof...(Ts);
  };

  template<class T, class... Ts>
  struct type_id {
    static constexpr int index = 0;
    static_assert(!ISTYPE(T, T), "Unkown type. ");
  };

  template<class T, class... Ts>
  struct type_id<T, type_list<T, Ts...>> {
    static constexpr int index = 0;
  };

  template<class T, class U, class... Ts>
  struct type_id<T, type_list<U, Ts...>> {
    static constexpr int index = 1 + type_id<T, type_list<Ts...>>::index;
  };

  template<class T, class... Ts>
  struct has {
    static constexpr bool has_type = false;
  };

  template<class T, class U, class... Ts>
  struct has<T, type_list<U, Ts...>> {
    static constexpr bool has_type = ISTYPE(T, U) || has<T, type_list<Ts...>>::has_type;
  };

}  // namespace core
#endif  // TYPE_LIST_H
