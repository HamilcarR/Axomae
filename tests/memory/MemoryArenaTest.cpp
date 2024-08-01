#include "MemoryArena.h"
#include "Test.h"
#include "math_random.h"
#include <cstdint>

class TestObject1 {
 public:
  std::string toString() { return "TestObject1"; }
};

TEST(MemoryArenaTest, construct) {
  core::memory::Arena<> allocator(65556);
  auto *buffer = allocator.construct<TestObject1>(10);
  for (int i = 0; i < 10; i++)
    ASSERT_EQ(buffer[i].toString(), "TestObject1");
}

TEST(MemoryArenaTest, alloc) {
  core::memory::Arena<> allocator;
  /* check 16 bytes alignment addresses*/
  for (int i = 0; i < 1024; i++) {
    auto ptr = reinterpret_cast<uintptr_t>(allocator.alloc(math::random::nrandi(1, 10000)));
    ASSERT_EQ(ptr % 0x10, 0);
  }
}
