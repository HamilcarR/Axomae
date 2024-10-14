#include "internal/memory/MemoryArena.h"
#include "Test.h"
#include "internal/common/math/math_random.h"
#include <cstdint>
#include <gtest/gtest.h>

class TestObject1 {
 public:
  std::string toString() { return "TestObject1"; }
};

TEST(MemoryArenaTest, construct) {
  core::memory::MemoryArena<> allocator(65556);
  auto *buffer = allocator.construct<TestObject1>(10);
  for (int i = 0; i < 10; i++)
    ASSERT_EQ(buffer[i].toString(), "TestObject1");
}

TEST(MemoryArenaTest, alloc) {
  core::memory::MemoryArena<> allocator;
  /* check 16 bytes alignment addresses*/
  for (int i = 0; i < 1024; i++) {
    void *ptr = allocator.allocate(math::random::nrandi(1, 10000));
    ASSERT_EQ(reinterpret_cast<uintptr_t>(ptr) % 0x10, 0);
  }
}

TEST(MemoryArenaTest, getTotalSize) {
  std::size_t default_block_size = core::memory::DEFAULT_BLOCK_SIZE;
  core::memory::MemoryArena<> allocator(default_block_size);
  ASSERT_EQ(allocator.getTotalSize(), 0);
  allocator.allocate(default_block_size);
  ASSERT_EQ(allocator.getTotalSize(), default_block_size);
  std::size_t align_size;
  /* check size alignment on 16 bytes*/
  for (int i = 0; i < 200; i++) {
    align_size = math::random::nrandi(1, core::memory::DEFAULT_BLOCK_SIZE);
    allocator.allocate(default_block_size + align_size);
    ASSERT_EQ(allocator.getTotalSize() % 0x10, 0);
  }
}

TEST(MemoryArenaTest, reset) {
  core::memory::MemoryArena<> allocator;
  allocator.allocate(100000);
  allocator.allocate(20303030);
  allocator.allocate(404040239);
  ASSERT_EQ(allocator.getUsedBlocksNum(), 2);
  ASSERT_EQ(allocator.getFreeBlocksNum(), 0);
  allocator.reset();
  ASSERT_EQ(allocator.getUsedBlocksNum(), 0);
  ASSERT_EQ(allocator.getFreeBlocksNum(), 2);
}

TEST(MemoryArenaTest, deallocate) {
  core::memory::MemoryArena<> allocator;
  void *ptr = allocator.allocate(1000000);
  allocator.deallocate(ptr);
  ASSERT_EQ(allocator.getUsedBlocksNum(), 0);
  ASSERT_EQ(allocator.getFreeBlocksNum(), 1);
  ptr = allocator.allocate(999);
  ASSERT_EQ(allocator.getFreeBlocksNum(), 0);
  allocator.deallocate(ptr);
}