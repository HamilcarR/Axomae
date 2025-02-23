#include <internal/memory/tag_ptr.h>
#include <unit_test/Test.h>
#include <unit_test/corelib/internal/common/type_listTest.h>

template<class T, class... Ts>
void check_all_tag() {
  if constexpr (sizeof...(Ts)) {
    T temp;
    using pack = core::type_list<PACKTYPES>;
    core::tag_ptr<PACKTYPES> ptr(&temp);
    int type_index = core::type_id<T, pack>::index;
    ASSERT_EQ(type_index, ptr.tag() - 1);
  }
}

TEST(tag_ptr_test, tag) { check_all_tag<PACKTYPES>(); }

template<class T, class... Ts>
void check_all_isType() {
  if constexpr (sizeof...(Ts) > 0) {
    T temp_obj;
    core::tag_ptr<PACKTYPES> ptr(&temp_obj);
    ASSERT_TRUE(ptr.isType<T>());
    check_all_isType<Ts...>();
  }
}

TEST(tag_ptr_test, isType) { check_all_isType<PACKTYPES>(); }

template<class T, class... Ts>
void check_deref() {
  using Tag = core::tag_ptr<PACKTYPES>;
  if constexpr (sizeof...(Ts) > 0) {
    std::unique_ptr<T> var = std::make_unique<T>();
    Tag ptr = var.get();
    ASSERT_EQ(var.get(), ptr.get<T>());
    ASSERT_EQ(var.get(), ptr.get());
  }
}

TEST(tag_ptr_test, get) { check_deref<PACKTYPES>(); }

class TestClass1 {
 public:
  std::string toString() { return "TestClass1"; }
  int getNum(int n) { return n + 1; }
};
class TestClass2 {
 public:
  std::string toString() { return "TestClass2"; }
  int getNum(int n) { return n + 2; }
};
class TestClass3 {
 public:
  std::string toString() { return "TestClass3"; }
  int getNum(int n) { return n + 3; }
};

class DispatchTest : public core::tag_ptr<TestClass1, TestClass2, TestClass3> {
 public:
  using tag_ptr::tag_ptr;

  ax_no_discard std::string toString() {
    auto d = [](auto ptr) { return ptr->toString(); };
    return host_dispatch(d);
  }
};

TEST(tag_ptr_test, host_dispatch) {
  TestClass1 test1;
  DispatchTest dt = &test1;
  ASSERT_TRUE(dt.toString() == "TestClass1");
  ASSERT_FALSE(dt.toString() == "TestClass2");
  ASSERT_FALSE(dt.toString() == "TestClass3");

  TestClass2 test2;
  dt = &test2;
  ASSERT_FALSE(dt.toString() == "TestClass1");
  ASSERT_TRUE(dt.toString() == "TestClass2");
  ASSERT_FALSE(dt.toString() == "TestClass3");

  TestClass3 test3;
  dt = &test3;
  ASSERT_FALSE(dt.toString() == "TestClass1");
  ASSERT_FALSE(dt.toString() == "TestClass2");
  ASSERT_TRUE(dt.toString() == "TestClass3");
}

/* Test if tagged pointers are read from the device side when pinning memory */

#if defined(AXOMAE_USE_CUDA)
#  include "cuda/tag_device_test.cuh"

TEST(tag_ptr_test, device_share) { tag_ptr_kernel_test(); }

#endif
