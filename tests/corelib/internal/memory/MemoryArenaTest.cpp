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
  core::memory::ByteArena allocator(65556);
  auto *buffer = allocator.construct<TestObject1>(10);
  for (int i = 0; i < 10; i++)
    ASSERT_EQ(buffer[i].toString(), "TestObject1");

  auto address = reinterpret_cast<std::uintptr_t>(allocator.allocate(65556, "", core::memory::B128_ALIGN));
  ASSERT_EQ(address % core::memory::B128_ALIGN, 0);
}

TEST(MemoryArenaTest, alloc) {
  core::memory::MemoryArena<> allocator;
  /* check 16 bytes alignment addresses*/
  math::random::CPUPseudoRandomGenerator generator;
  for (int i = 0; i < 1024; i++) {
    void *ptr = allocator.allocate(generator.nrandi(1, 10000));
    ASSERT_EQ(reinterpret_cast<uintptr_t>(ptr) % allocator.getCurrentAlignment(), 0);
  }
}

TEST(MemoryArenaTest, getTotalSize) {
  std::size_t default_block_size = core::memory::DEFAULT_BLOCK_SIZE;
  core::memory::MemoryArena<> allocator(default_block_size);
  ASSERT_EQ(allocator.getTotalSize(), 0);
  allocator.allocate(default_block_size);
  ASSERT_EQ(allocator.getTotalSize(), default_block_size);
  std::size_t align_size{};

  math::random::CPUPseudoRandomGenerator generator;
  /* check platform alignment. */
  for (int i = 0; i < 200; i++) {
    align_size = generator.nrandi(1, core::memory::DEFAULT_BLOCK_SIZE);
    auto aligned_ptr = reinterpret_cast<uintptr_t>(allocator.allocate(default_block_size + align_size), core::memory::PLATFORM_ALIGN);
    ASSERT_EQ(aligned_ptr % core::memory::PLATFORM_ALIGN, 0);
    ASSERT_EQ(allocator.getTotalSize() % allocator.getCurrentAlignment(), 0);
  }

  /* 128 bytes alignment.*/
  for (int i = 0; i < 200; i++) {
    align_size = generator.nrandi(1, core::memory::DEFAULT_BLOCK_SIZE);
    auto aligned_ptr = reinterpret_cast<uintptr_t>(allocator.allocate(default_block_size + align_size), core::memory::B128_ALIGN);
    ASSERT_EQ(aligned_ptr % core::memory::B128_ALIGN, 0);
    ASSERT_EQ(allocator.getTotalSize() % allocator.getCurrentAlignment(), 0);
  }

  /* 64 bytes alignment. */
  for (int i = 0; i < 200; i++) {
    align_size = generator.nrandi(1, core::memory::DEFAULT_BLOCK_SIZE);
    auto aligned_ptr = reinterpret_cast<uintptr_t>(allocator.allocate(default_block_size + align_size), core::memory::B64_ALIGN);
    ASSERT_EQ(aligned_ptr % core::memory::B64_ALIGN, 0);
    ASSERT_EQ(allocator.getTotalSize() % allocator.getCurrentAlignment(), 0);
  }
  /* 32 bytes alignment.*/
  for (int i = 0; i < 200; i++) {
    align_size = generator.nrandi(1, core::memory::DEFAULT_BLOCK_SIZE);
    auto aligned_ptr = reinterpret_cast<uintptr_t>(allocator.allocate(default_block_size + align_size), core::memory::B32_ALIGN);
    ASSERT_EQ(aligned_ptr % core::memory::B32_ALIGN, 0);
    ASSERT_EQ(allocator.getTotalSize() % allocator.getCurrentAlignment(), 0);
  }

  /* 16 bytes alignment.*/
  for (int i = 0; i < 200; i++) {
    align_size = generator.nrandi(1, core::memory::DEFAULT_BLOCK_SIZE);
    auto aligned_ptr = reinterpret_cast<uintptr_t>(allocator.allocate(default_block_size + align_size), core::memory::B16_ALIGN);
    ASSERT_EQ(aligned_ptr % core::memory::B16_ALIGN, 0);
    ASSERT_EQ(allocator.getTotalSize() % allocator.getCurrentAlignment(), 0);
  }

  /* 8 bytes alignment.*/
  for (int i = 0; i < 200; i++) {
    align_size = generator.nrandi(1, core::memory::DEFAULT_BLOCK_SIZE);
    auto aligned_ptr = reinterpret_cast<uintptr_t>(allocator.allocate(default_block_size + align_size), core::memory::B8_ALIGN);
    ASSERT_EQ(aligned_ptr % core::memory::B8_ALIGN, 0);
    ASSERT_EQ(allocator.getTotalSize() % allocator.getCurrentAlignment(), 0);
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

TEST(MemoryArenaTest, copyRange) {
  core::memory::ByteArena arena;
  void *pool = arena.allocate(256, "", core::memory::PLATFORM_ALIGN);
  std::array<uint8_t, 128> array1, array2;
  for (uint32_t i = 0; i < 256; i++) {
    if (i < 128)
      array1[i] = i;
    else
      array2[i - 128] = i;
  }
  uint8_t *ptr_array1 = static_cast<uint8_t *>(arena.copyRange(array1.data(), pool, 128, 0));
  ASSERT_EQ(reinterpret_cast<std::uintptr_t>(ptr_array1) % core::memory::PLATFORM_ALIGN, 0);
  for (uint32_t i = 0; i < array1.size(); i++) {
    EXPECT_EQ(array1[i], ptr_array1[i]);
  }
  uint8_t *ptr_array2 = static_cast<uint8_t *>(arena.copyRange(array2.data(), pool, 128, 1));
  ASSERT_EQ(reinterpret_cast<std::uintptr_t>(ptr_array2) % core::memory::PLATFORM_ALIGN, 0);
  for (uint32_t i = 0; i < array2.size(); i++) {
    EXPECT_EQ(array2[i], ptr_array2[i]);
  }
}