#include "DatabaseBuilderTest.h"
#include "RenderingDatabaseInterface.h"
#include "Test.h"

/**
 * Tests the caching methods of IResourceDB
 */

class TestString {
 public:
  std::string str;

 public:
  TestString() = default;
  explicit TestString(const std::string &id) : str(id) {}
  explicit TestString(const char *id) : str(id != nullptr ? id : "") {}

  bool operator==(const std::string &cmp) const { return str == cmp; }
};

constexpr uint ALLOC_SIZE = 20048 * sizeof(TestString);
constexpr uint NUM_TESTS = 200;

class TestDatabase : public IntegerResourceDB<TestString> {
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
    EXPECT_NE(ptr, nullptr);
    EXPECT_EQ(arena.getTotalSize(), test_database.getCacheSize(ptr));
  }
}

TEST(IResourceDB, getCacheSize) {
  core::memory::MemoryArena arena;
  TestDatabase test_database(&arena);
  ASSERT_EQ(test_database.getCacheSize(nullptr), DATABASE_CACHE_INVALID_SIZE);
  for (uint32_t i = 0; i < NUM_TESTS; i++) {
    uint8_t *ptr = reinterpret_cast<uint8_t *>(math::random::nrandi(1, 65556));
    std::size_t size = test_database.getCacheSize(ptr);
    ASSERT_EQ(size, DATABASE_CACHE_INVALID_SIZE);
  }
  for (uint32_t i = 0; i < NUM_TESTS; i++) {
    std::size_t block_size_arena = math::random::nrandi(2048, 65556);
    uint8_t *ptr = nullptr;
    ASSERT_NE(ptr = test_database.reserveCache(block_size_arena), nullptr);
    EXPECT_EQ(block_size_arena, test_database.getCacheSize(ptr));
  }
}

TEST(IResourceDB, invalidateCaches) {
  core::memory::MemoryArena arena;
  TestDatabase test_database(&arena);
  for (uint32_t i = 0; i < NUM_TESTS; i++) {
    std::size_t block_size_arena = math::random::nrandi(2048, 65556);
    ASSERT_NE(test_database.reserveCache(block_size_arena), nullptr);
  }
  ASSERT_NE(arena.getUsedBlocksNum(), 0);
  test_database.invalidateCaches();
  ASSERT_EQ(arena.getUsedBlocksNum(), 0);
  ASSERT_NE(arena.getFreeBlocksNum(), 0);
}

static void test_no_cache() {
  TestDatabase test_database;
  database::Result result = test_database.addCached<TestString>(false, nullptr);
  ASSERT_EQ(result.object, nullptr);
}

static void verify_objects_memory(uint8_t *cache, uint32_t total_size, const std::string &base_str) {
  for (uint32_t i = 0; i < total_size; i++) {
    std::size_t offset = i * sizeof(TestString);
    auto current_ptr = reinterpret_cast<TestString *>(cache + offset);
    EXPECT_EQ(current_ptr->str, base_str + std::to_string(i));
  }
}

static void verify_out_of_bounds() {
  core::memory::MemoryArena arena;
  TestDatabase test_database(&arena);
  uint8_t *cache = test_database.reserveCache(sizeof(TestString));
  const std::string valid = "Not out-of-bounds";
  database::Result result = test_database.addCached<TestString>(false, cache, valid);
  ASSERT_NE(result.object, nullptr);
  EXPECT_EQ(*result.object, valid);
  const std::string invalid = "Out-of-bounds";
  result = test_database.addCached<TestString>(false, cache, invalid);
  database::Result<int, TestString> empty{};
  EXPECT_EQ(result, empty);
}

void add_string() {
  core::memory::MemoryArena arena;
  TestDatabase test_database(&arena);
  uint8_t *cache = test_database.reserveCache(ALLOC_SIZE);
  std::vector<database::Result<int, TestString>> results;
  const std::string base_str_test = "Test addCached";
  for (uint32_t i = 0; i < NUM_TESTS; i++) {
    const std::string test = base_str_test + std::to_string(i);
    database::Result result = test_database.addCached<TestString>(false, cache, test);
    ASSERT_NE(result.object, nullptr);
    EXPECT_EQ(*result.object, test);
  }
  verify_objects_memory(cache, NUM_TESTS, base_str_test);
  verify_out_of_bounds();
}

static void add_texture() {
  core::memory::ByteArena arena;
  TestDatabase test_database(&arena);
}

static void test_add_to_cache() {
  add_string();
  add_texture();
}

TEST(IResourceDB, addCached) {
  test_no_cache();
  test_add_to_cache();
}