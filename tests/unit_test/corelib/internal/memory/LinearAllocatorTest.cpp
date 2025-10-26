#include <gtest/gtest.h>
#include <internal/common/math/math_random.h>
#include <internal/memory/Allocator.h>
#include <unit_test/Test.h>

class TestClass {
  unsigned position;

 public:
  TestClass() = default;
  TestClass(unsigned p) : position(p) {}
  unsigned print() const { return position; }
};

class TestClass1b {
  uint8_t position;

 public:
  TestClass1b() = default;
  TestClass1b(uint8_t p) : position(p) {}
  uint8_t print() const { return position; }
};

void test1() {

  axstd::StaticAllocator64kb allocator;
  TestClass *zero = allocator.construct<TestClass>(0);
  TestClass *one = allocator.construct<TestClass>(1);
  TestClass *two = allocator.construct<TestClass>(2);
  TestClass *three = allocator.construct<TestClass>(3);

  ASSERT_EQ(zero->print(), 0);
  ASSERT_EQ(one->print(), 1);
  ASSERT_EQ(two->print(), 2);
  ASSERT_EQ(three->print(), 3);
}

void test2() {

  axstd::StaticAllocator64kb allocator;
  TestClass1b *zero = allocator.construct<TestClass1b>(0);
  TestClass1b *one = allocator.construct<TestClass1b>(1);
  TestClass1b *two = allocator.construct<TestClass1b>(2);
  TestClass1b *three = allocator.construct<TestClass1b>(3);

  ASSERT_EQ(zero->print(), 0);
  ASSERT_EQ(one->print(), 1);
  ASSERT_EQ(two->print(), 2);
  ASSERT_EQ(three->print(), 3);
}

void test_bound() {
  axstd::StaticAllocator16b allocator;
  struct priv {
    alignas(4) uint8_t _pad[4];
  };
  for (unsigned i = 0; i < 4; i++) {
    ASSERT_NE(allocator.construct<priv>(), nullptr);
  }
  ASSERT_EQ(allocator.construct<priv>(), nullptr);
}

TEST(LinearAllocatorTest, allocate) {
  test1();
  test2();
  test_bound();
}
