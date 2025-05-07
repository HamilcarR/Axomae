#include "texturing/texture_interop_storage.h"
#include "Test.h"
#include "texturing/texture_datastructures.h"
#include <algorithm>
#include <gtest/gtest.h>
#include <internal/common/axstd/span.h>
#include <string>

namespace nvt = nova::texturing;

template<unsigned NUM_COLLECTIONS>
class IdRemovalTrackerTest : public nvt::IdRemovalTracker<NUM_COLLECTIONS> {

 public:
  template<unsigned NUM_COLLECTION>
  axstd::span<const std::size_t> get() const {
    return nvt::IdRemovalTracker<NUM_COLLECTIONS>::empty_ids[NUM_COLLECTION];
  }
};

class DeviceDummyBuffer : nvt::DummyBufferTracker {};

template<class C>
bool find(const C &view, const std::size_t element) {
  return std::find_if(view.begin(), view.end(), [&element](const std::size_t &test) { return element == test; }) != view.end();
}

class IdAppendTest : public IdRemovalTrackerTest<3> {
  enum IDTYPE : unsigned { U32 = 0, F32 = 1, DB = 2 };
  std::vector<F32Texture> collection0;
  std::vector<U32Texture> collection1;
  std::vector<DeviceDummyBuffer> collection2;

 public:
  void add(const F32Texture &tex) { collection0.push_back(tex); }
  void add(const DeviceDummyBuffer &tex) { collection2.push_back(tex); }
  void add(const U32Texture &tex) { collection1.push_back(tex); }

  void removeF32(std::size_t index) { disableID<F32>(index); }
  void removeU32(std::size_t index) { disableID<U32>(index); }
  void removeDb(std::size_t index) { disableID<DB>(index); }

  bool existIdF32(std::size_t index) {
    axstd::span<const std::size_t> f32view = get<F32>();
    return find(f32view, index);
  }
  bool existIdU32(std::size_t index) {
    axstd::span<const std::size_t> u32view = get<U32>();
    return find(u32view, index);
  }
  bool existIdDB(std::size_t index) {
    axstd::span<const std::size_t> DBview = get<DB>();
    return find(DBview, index);
  }
};

TEST(IdRemovalTrackerTest, disableID) {
  IdAppendTest test1;
  for (int i = 0; i < 5; i++) {
    test1.add(F32Texture());
    test1.add(U32Texture());
    test1.add(DeviceDummyBuffer());
  }

  test1.removeF32(0);
  ASSERT_TRUE(test1.existIdF32(0));

  test1.removeU32(3);
  ASSERT_TRUE(test1.existIdU32(3));

  test1.removeDb(4);
  ASSERT_TRUE(test1.existIdDB(4));
}

class HostPolicyTest : public nvt::HostPolicy {};

TEST(texture_interop_storage_HostPolicyTest, add) {
  HostPolicyTest test;
  std::vector<std::size_t> u32_indices;
  std::vector<std::size_t> f32_indices;

  for (int i = 0; i < 5; i++) {
    u32_indices.push_back(test.add(U32Texture()));
    f32_indices.push_back(test.add(F32Texture()));
  }

  for (int i = 0; i < 5; i++) {
    ASSERT_TRUE(find(u32_indices, i));
    ASSERT_TRUE(find(f32_indices, i));
  }
  std::vector<std::size_t> f32_free_ids;

  test.removeF32(0);
  test.removeF32(1);
  test.removeF32(2);
  f32_free_ids.push_back(test.add(F32Texture()));
  f32_free_ids.push_back(test.add(F32Texture()));
  f32_free_ids.push_back(test.add(F32Texture()));
  ASSERT_TRUE(find(f32_free_ids, 0));
  ASSERT_TRUE(find(f32_free_ids, 1));
  ASSERT_TRUE(find(f32_free_ids, 2));

  std::vector<std::size_t> u32_free_ids;

  test.removeU32(0);
  test.removeU32(1);
  test.removeU32(2);
  u32_free_ids.push_back(test.add(U32Texture()));
  u32_free_ids.push_back(test.add(U32Texture()));
  u32_free_ids.push_back(test.add(U32Texture()));
  ASSERT_TRUE(find(u32_free_ids, 0));
  ASSERT_TRUE(find(u32_free_ids, 1));
  ASSERT_TRUE(find(u32_free_ids, 2));
}
