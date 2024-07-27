#ifndef TYPE_LIST_H
#define TYPE_LIST_H
#include "project_macros.h"

/* maybe rename to type.h*/

namespace core {
  template<class... Ts>
  struct type_list {
    static constexpr int size = sizeof...(Ts);
  };

  template<class T, class... Ts>
  struct type_id {
    static constexpr int index = 0;
    static_assert(!ISTYPE(T, T), "Unkown type. :");
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

  template<class... Ts>
  struct is_same_type;

  template<>
  struct is_same_type<> {
    static constexpr bool same = true;
  };

  template<class T>
  struct is_same_type<T> {
    static constexpr bool same = true;
  };

  template<class T, class U, class... Ts>
  struct is_same_type<T, U, Ts...> {
    static constexpr bool same = ISTYPE(T, U) && is_same_type<U, Ts...>::same;
  };

  template<class... Ts>
  struct same_type;

  template<class T, class... Ts>
  struct same_type<T, Ts...> {
    using type = T;
    static_assert(is_same_type<T, Ts...>::same, "Types not similar.");
  };

  template<class F, class... Args>
  struct return_type {
    using RETURN_TYPE = typename same_type<typename std::invoke_result_t<F, Args *>...>::type;
  };

  template<class F, class... Args>
  struct return_const_type {
    using RETURN_TYPE = typename same_type<typename std::invoke_result_t<F, const Args *>...>::type;
  };
}  // namespace core
#endif  // TYPE_LIST_H
