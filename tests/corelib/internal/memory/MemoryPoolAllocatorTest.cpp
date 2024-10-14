#include "internal/memory/MemoryPoolAllocator.h"
#include "Test.h"
#include "internal/memory/MemoryArena.h"

#include <internal/common/math/math_random.h>
namespace memory = core::memory;

constexpr int NUM_TESTS = 100;
constexpr std::size_t MAX_ALLOCATION_SIZE = 1 << 16;
TEST(MemoryPoolAllocatorTest, constructor) {
  /* Arena non initialized */
  ASSERT_THROW(memory::MemoryPoolAllocator<int>(nullptr), memory::exception::ArenaInvalidStateException);
  auto arena = memory::MemoryArena();
  ASSERT_NO_THROW(auto v = core::memory::MemoryPoolAllocator<int>(&arena));
}

TEST(MemoryPoolAllocatorTest, allocate) {
  auto arena = memory::MemoryArena<>();
  core::memory::MemoryPoolAllocator<int> custom_alloc;
  ASSERT_NO_THROW(custom_alloc = core::memory::MemoryPoolAllocator<int>(&arena));

  int *ptr = custom_alloc.allocate(1);
  ASSERT_NE(ptr, nullptr);
  ptr = custom_alloc.allocate(66666);
  ASSERT_NE(ptr, nullptr);
  /* Test alloc exception at size limits */
  ASSERT_THROW(ptr = custom_alloc.allocate(std::numeric_limits<std::size_t>::max()), std::bad_alloc);
}

template<class T>
void test_fill_buffer(T *buffer, std::size_t size) {
  for (std::size_t i = 0; i < size; i++) {
    buffer[i] = math::random::nrandi(0, 200);
  }
}

TEST(MemoryPoolAllocatorTest, deallocate) {
  auto arena = memory::MemoryArena<>();
  core::memory::MemoryPoolAllocator<int> custom_alloc;
  ASSERT_THROW(custom_alloc.deallocate(nullptr), memory::exception::ArenaInvalidStateException);
  for (int i = 0; i < NUM_TESTS; i++) {
    custom_alloc = core::memory::MemoryPoolAllocator<int>(&arena);
    int *buffer = nullptr;
    std::size_t buffer_size = math::random::nrandi(0, MAX_ALLOCATION_SIZE);
    ASSERT_NO_THROW(buffer = custom_alloc.allocate(buffer_size));
    test_fill_buffer(buffer, buffer_size);
    ASSERT_NO_THROW(custom_alloc.deallocate(buffer));
  }
}