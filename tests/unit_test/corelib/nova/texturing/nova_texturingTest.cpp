#include "texturing/nova_texturing.h"
#include "Test.h"
#include "texturing/NovaTextureInterface.h"
#include <algorithm>

namespace nvt = nova::texturing;

class TestImageTexture : public nvt::ImageTexture<uint32_t> {};
class TestConstantTexture : public nvt::ConstantTexture {};
class TestEnvmapTexture : public nvt::EnvmapTexture {};

static const int NUM_ALLOCS = 50;

TEST(TextureStorageTest, alloc) {
  nvt::TextureStorage storage;
  storage.allocConstant(NUM_ALLOCS);
  EXPECT_TRUE(storage.isConstantInit());
  EXPECT_TRUE(storage.isConstantEmpty());

  storage.allocImage(NUM_ALLOCS);
  EXPECT_TRUE(storage.isImageInit());
  EXPECT_TRUE(storage.isImageEmpty());

  storage.allocEnvmap(NUM_ALLOCS);
  EXPECT_TRUE(storage.isEnvmapInit());
  EXPECT_TRUE(storage.isEnvmapEmpty());
}

TEST(TextureStorageTest, add) {
  nvt::TextureStorage storage;
  storage.allocConstant(NUM_ALLOCS);
  std::size_t total_size = 0;
  for (int i = 1; i < NUM_ALLOCS; i++) {
    TestConstantTexture img;
    storage.add(img);
    EXPECT_EQ(i, storage.sizeConstant());
  }
  total_size += storage.sizeConstant();
  ASSERT_EQ(total_size, storage.pointers().size());

  storage.allocImage(NUM_ALLOCS);
  for (int i = 1; i < NUM_ALLOCS; i++) {
    TestImageTexture img;
    storage.add(img);
    EXPECT_EQ(i, storage.sizeImage());
  }
  total_size += storage.sizeImage();
  ASSERT_EQ(total_size, storage.pointers().size());

  storage.allocEnvmap(NUM_ALLOCS);
  for (int i = 1; i < NUM_ALLOCS; i++) {
    TestEnvmapTexture img;
    storage.add(img);
    EXPECT_EQ(i, storage.sizeEnvmap());
  }
  total_size += storage.sizeEnvmap();
  ASSERT_EQ(total_size, storage.pointers().size());
}

bool interfaces_cleared(const IntfTexCollection &storage_textures_interfaces, const std::vector<nvt::NovaTextureInterface> &test_case) {
  for (const auto &elem : test_case) {
    auto iterator = std::find_if(storage_textures_interfaces.begin(),
                                 storage_textures_interfaces.end(),
                                 [&elem](const nvt::NovaTextureInterface &compare) { return compare.get() == elem.get(); });
    if (iterator != storage_textures_interfaces.end())
      return false;
  }
  return true;
}

TEST(TextureStorageTest, clearElement) {
  nvt::TextureStorage storage;
  storage.allocConstant(NUM_ALLOCS);
  storage.allocEnvmap(NUM_ALLOCS);
  storage.allocImage(NUM_ALLOCS);
  std::vector<nvt::NovaTextureInterface> envmap_interfaces;
  std::vector<nvt::NovaTextureInterface> constant_interfaces;
  std::vector<nvt::NovaTextureInterface> image_interfaces;
  for (int i = 0; i < 20; i++) {
    TestImageTexture img;
    nvt::NovaTextureInterface base = storage.add(img);
    image_interfaces.push_back(base);

    TestEnvmapTexture env;
    base = storage.add(env);
    envmap_interfaces.push_back(base);

    TestConstantTexture cst;
    base = storage.add(cst);
    constant_interfaces.push_back(base);
  }

  storage.clearImage();
  EXPECT_TRUE(storage.images().empty());
  EXPECT_TRUE(interfaces_cleared(storage.pointers(), image_interfaces));

  storage.clearEnvmap();
  EXPECT_TRUE(storage.envmaps().empty());
  EXPECT_TRUE(interfaces_cleared(storage.pointers(), envmap_interfaces));

  storage.clearConstant();
  EXPECT_TRUE(storage.constants().empty());
  EXPECT_TRUE(interfaces_cleared(storage.pointers(), constant_interfaces));
}
