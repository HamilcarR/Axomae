#include "internal/common/type_list.h"
#include "Test.h"
#include "type_listTest.h"

template<class T, class... Ts>
constexpr void check_true(int i) {
  using pack = core::type_list<PACKTYPES>;
  constexpr int idx = core::type_id<T, pack>::index;
  ASSERT_EQ(idx, i);
  if constexpr (sizeof...(Ts) > 0)
    check_true<Ts...>(i + 1);
}

template<class T, class... Ts>
constexpr void check_false(int i) {
  using pack = core::type_list<PACKTYPES>;
  constexpr int idx = core::type_id<T, pack>::index;
  ASSERT_NE(idx, i);
  if constexpr (sizeof...(Ts) > 0)
    check_false<Ts...>(i + 1);
}

TEST(type_pack_test, type_id) {
  check_true<PACKTYPES>(0);
  check_false<void *, PACKTYPES>(0);
}

template<class T, class... Ts>
constexpr bool has_all() {
  using pack = core::type_list<PACKTYPES>;
  constexpr bool has = core::has<T, pack>::has_type;
  if constexpr (!has)
    return false;
  if constexpr (sizeof...(Ts) > 0)
    return has & has_all<Ts...>();
  return true;
}

TEST(type_pack_test, has_type) {
  constexpr bool has_a = has_all<PACKTYPES>();
  ASSERT_TRUE(has_a);
  constexpr bool has_no = has_all<PACKTYPES, uint64_t>();
  ASSERT_FALSE(has_no);
}