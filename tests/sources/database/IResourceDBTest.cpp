#include "DatabaseBuilderTest.h"
#include "RenderingDatabaseInterface.h"
#include "Test.h"

/**
 * This tests the caching methods of IResourceDB
 */

struct TestData {
  char test_data_id[64];
};

constexpr uint ALLOC_SIZE = 20048 * sizeof(TestData);
constexpr uint NUM_TESTS = 200;
class TestDatabase : public IntegerResourceDB<TestData> {
 public:
  CLASS_OCM(TestDatabase)
  explicit TestDatabase(core::memory::MemoryArena<std::byte> *arena) { setUpCacheMemory(arena); }
};

TEST(IResourceDB, reserveCache) {
  for (uint32_t test_i = 0; test_i < NUM_TESTS; test_i++) {
    std::size_t block_size_arena = math::random::nrandi(2048, 65556);
    core::memory::MemoryArena arena(block_size_arena);
    TestDatabase test_database(&arena);
    auto ptr = test_database.reserveCache(ALLOC_SIZE);
    EXPECT_EQ(arena.getTotalSize(), test_database.getCacheSize(ptr));
  }
}