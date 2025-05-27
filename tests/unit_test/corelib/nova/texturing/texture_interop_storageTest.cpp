#include "texturing/texture_interop_storage.h"
#include "Test.h"
#include "texturing/texture_datastructures.h"
#include <algorithm>
#include <gtest/gtest.h>
#include <internal/common/axstd/span.h>
#include <internal/common/math/math_random.h>
#include <internal/device/gpgpu/device_resource_descriptors.h>
#include <internal/device/gpgpu/device_texture_descriptors.h>
#include <string>

namespace nvt = nova::texturing;

template<class T>
class IdRemovalTrackerTest : public nvt::IdRemovalTracker<T> {

 public:
  axstd::span<const std::size_t> get() const { return nvt::IdRemovalTracker<T>::empty_ids; }
};

class DeviceDummyBuffer : nvt::DummyBufferTracker {};

template<class C>
bool find(const C &view, const std::size_t element) {
  return std::find_if(view.begin(), view.end(), [&element](const std::size_t &test) { return element == test; }) != view.end();
}

class IdAppendTest {

  std::vector<F32Texture> collection0;
  IdRemovalTrackerTest<F32Texture> f32_remover;

  std::vector<U32Texture> collection1;
  IdRemovalTrackerTest<U32Texture> u32_remover;

  std::vector<DeviceDummyBuffer> collection2;
  IdRemovalTrackerTest<DeviceDummyBuffer> db_remover;

 public:
  void add(const F32Texture &tex) { collection0.push_back(tex); }
  void add(const DeviceDummyBuffer &tex) { collection2.push_back(tex); }
  void add(const U32Texture &tex) { collection1.push_back(tex); }

  void removeF32(std::size_t index) { f32_remover.disableID(index); }
  void removeU32(std::size_t index) { u32_remover.disableID(index); }
  void removeDb(std::size_t index) { db_remover.disableID(index); }

  bool existIdF32(std::size_t index) {
    axstd::span<const std::size_t> f32view = f32_remover.get();
    return find(f32view, index);
  }
  bool existIdU32(std::size_t index) {
    axstd::span<const std::size_t> u32view = u32_remover.get();
    return find(u32view, index);
  }
  bool existIdDB(std::size_t index) {
    axstd::span<const std::size_t> DBview = db_remover.get();
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

TEST(texture_interop_storage_HostPolicyTest, addAndRemove) {
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

class DummyDeviceBuffer {
  bool is_mapped{false};

 public:
  DummyDeviceBuffer(GLuint, GLenum, device::gpgpu::ACCESS_TYPE) {}

  bool isMapped() const { return is_mapped; }
  void map() { is_mapped = true; }
  void unmap() { is_mapped = false; }
};

class DeviceStorageTestFake : public nvt::DeviceStorageInterface {
  std::vector<DummyDeviceBuffer> f32_dummy_buffers;
  nvt::IdRemovalTracker<DummyDeviceBuffer> f32_id_tracker;
  std::vector<device::gpgpu::APITextureHandle> f32_api_texture_handles;

  std::vector<DummyDeviceBuffer> u32_dummy_buffers;
  nvt::IdRemovalTracker<DummyDeviceBuffer> u32_id_tracker;
  std::vector<device::gpgpu::APITextureHandle> u32_api_texture_handles;

 public:
  std::size_t addF32(GLuint texture_id, GLenum format, device::gpgpu::ACCESS_TYPE access_type) override {
    f32_dummy_buffers.emplace_back(texture_id, format, access_type);
    return f32_dummy_buffers.size() - 1;
  }
  std::size_t addU32(GLuint texture_id, GLenum format, device::gpgpu::ACCESS_TYPE access_type) override {
    u32_dummy_buffers.emplace_back(texture_id, format, access_type);
    return u32_dummy_buffers.size() - 1;
  }

  void removeF32(std::size_t index) override { f32_id_tracker.disableID(index); }
  void removeU32(std::size_t index) override { u32_id_tracker.disableID(index); }

  void clearF32() override {
    f32_dummy_buffers.clear();
    f32_api_texture_handles.clear();
    f32_id_tracker.reset();
  }
  void clearU32() override {
    u32_dummy_buffers.clear();
    u32_api_texture_handles.clear();
    u32_id_tracker.reset();
  }

  IntopImgTexView u32Handles() const override { return u32_api_texture_handles; }
  IntopImgTexView f32Handles() const override { return f32_api_texture_handles; }

  void mapBuffers() override {
    f32_api_texture_handles.clear();
    for (const auto &elem : f32_dummy_buffers)
      f32_api_texture_handles.push_back(generate_dummy_handle());
    for (const auto &elem : u32_dummy_buffers)
      u32_api_texture_handles.push_back(generate_dummy_handle());
  }
  void mapResources() override {
    for (auto &elem : f32_dummy_buffers)
      elem.map();
    for (auto &elem : u32_dummy_buffers)
      elem.map();
  }

  void release() override {
    for (auto &elem : f32_dummy_buffers)
      elem.unmap();
    for (auto &elem : u32_dummy_buffers)
      elem.unmap();
  }

  bool isMapped() const {
    if (f32_dummy_buffers.empty() || u32_dummy_buffers.empty())
      return false;
    for (const auto &elem : f32_dummy_buffers)
      if (!elem.isMapped())
        return false;
    for (const auto &elem : u32_dummy_buffers)
      if (!elem.isMapped())
        return false;

    return true;
  }

 private:
  std::size_t generate_dummy_handle() const {
    math::random::CPUPseudoRandomGenerator gen;
    return (std::size_t)gen.nrandi(1, 100);
  }
};

class TextureReferenceStorageTest : public nvt::TextureReferencesStorage<nvt::DevicePolicy> {
 public:
  TextureReferenceStorageTest(std::unique_ptr<nvt::DeviceStorageInterface> dep) : TextureReferencesStorage(std::move(dep)) {}
};

static void test_empty_u32_views(const nvt::u32tex_shared_views_s &u32texview) { ASSERT_TRUE(u32texview.managed_tex_view.empty()); }

static void test_empty_f32_views(const nvt::f32tex_shared_views_s &f32texview) { ASSERT_TRUE(f32texview.managed_tex_view.empty()); }

static void test_notempty_u32_views(const nvt::u32tex_shared_views_s &u32texview) { ASSERT_FALSE(u32texview.managed_tex_view.empty()); }

static void test_notempty_f32_views(const nvt::f32tex_shared_views_s &f32texview) { ASSERT_FALSE(f32texview.managed_tex_view.empty()); }

static void test_notempty_u32_handle(const nvt::u32tex_shared_views_s &u32texview) { ASSERT_FALSE(u32texview.interop_handles.empty()); }

static void test_notempty_f32_handle(const nvt::f32tex_shared_views_s &f32texview) { ASSERT_FALSE(f32texview.interop_handles.empty()); }

static void test_empty_u32_handle(const nvt::u32tex_shared_views_s &u32texview) { ASSERT_TRUE(u32texview.interop_handles.empty()); }

static void test_empty_f32_handle(const nvt::f32tex_shared_views_s &f32texview) { ASSERT_TRUE(f32texview.interop_handles.empty()); }

TEST(TextureReferenceStorageTest, add) {
  TextureReferenceStorageTest test(std::make_unique<DeviceStorageTestFake>());

  test_empty_f32_views(test.getF32TexturesViews());
  test_empty_u32_views(test.getU32TexturesViews());

  test.add(F32Texture(), 0);
  test.add(U32Texture(), 1);

  test_notempty_f32_views(test.getF32TexturesViews());
  test_notempty_u32_views(test.getU32TexturesViews());
}

TEST(TextureReferenceStorageTest, mapBuffers) {
  TextureReferenceStorageTest test(std::make_unique<DeviceStorageTestFake>());

  test_empty_f32_handle(test.getF32TexturesViews());
  test_empty_u32_handle(test.getU32TexturesViews());

  test.mapBuffers();

  test_empty_f32_handle(test.getF32TexturesViews());
  test_empty_u32_handle(test.getU32TexturesViews());

  test.add(F32Texture(), 0);
  test.add(U32Texture(), 1);

  test.mapBuffers();

  test_notempty_f32_handle(test.getF32TexturesViews());
  test_notempty_u32_handle(test.getU32TexturesViews());
}

TEST(TextureReferenceStorageTest, mapResources) {
  std::unique_ptr<DeviceStorageTestFake> device_storage = std::make_unique<DeviceStorageTestFake>();
  DeviceStorageTestFake *ptr_dev_storage = device_storage.get();  //  we know that device_storage lifespan is the same as TextureReferenceStorageTest.
  TextureReferenceStorageTest test(std::move(device_storage));

  ASSERT_FALSE(ptr_dev_storage->isMapped());

  test.add(F32Texture(), 0);
  test.add(U32Texture(), 1);

  test.mapResources();

  ASSERT_TRUE(ptr_dev_storage->isMapped());

  test.release();

  ASSERT_FALSE(ptr_dev_storage->isMapped());
}

TEST(TextureReferenceStorageTest, remove) {
  std::unique_ptr<DeviceStorageTestFake> device_storage = std::make_unique<DeviceStorageTestFake>();
  TextureReferenceStorageTest test(std::move(device_storage));

  test.add(F32Texture(), 0);
  test.add(F32Texture(), 0);
  test.add(F32Texture(), 0);
  test.add(F32Texture(), 0);

  test.removeF32(2);

  ASSERT_EQ(test.add(F32Texture(), 0), 2);

  test.add(U32Texture(), 0);
  test.add(U32Texture(), 0);
  test.add(U32Texture(), 0);
  test.add(U32Texture(), 0);

  test.removeU32(3);

  ASSERT_EQ(test.add(U32Texture(), 0), 3);
}
