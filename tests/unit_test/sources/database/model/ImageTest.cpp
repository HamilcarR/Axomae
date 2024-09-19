#include "Image.h"
#include <internal/common/axstd/span.h>
#include <internal/common/math/math_utils.h>
#include <unit_test/Test.h>
constexpr int MAX_W = 3;
constexpr int MAX_H = 5;

template<class T>
class DataHostStorageTest : public ImageHolderDataStorageInterface<T> {
 private:
  std::vector<T> internal_data{};

 public:
  T *data() override { return internal_data.data(); }

  const T *data() const override { return internal_data.data(); }

  std::size_t size() const override { return internal_data.size(); }

  T &operator[](std::size_t index) override { return internal_data[index]; }

  const T &operator[](std::size_t index) const override { return internal_data[index]; }

  void reserve(std::size_t size) override { internal_data.reserve(size); }

  void resize(std::size_t size, const T &value) override { internal_data.resize(size, value); }

  void clear() override { internal_data.clear(); }
};

template<class T>
class ImageHolderTest : public image::ImageHolder<T> {
  using BASE = image::ImageHolder<T>;

 public:
  using BASE::BASE;
};

TEST(ImageTest, flip_v) {
  std::unique_ptr<ImageHolderDataStorageInterface<int>> storage_data = std::make_unique<DataHostStorageTest<int>>();
  ImageHolderTest<int> image(std::move(storage_data));
  image.data().resize(MAX_H * MAX_W * 4, 0);
  image.metadata.channels = 4;
  image.metadata.height = MAX_H;
  image.metadata.width = MAX_W;

  for (int i = 0; i < MAX_H * MAX_W * 4; i++)
    image.data()[i] = i;
  std::vector<int> ordered(image.data().begin(), image.data().end());
  image.flip_v();
  for (int i = 0; i < MAX_H; i++)
    for (int j = 0; j < MAX_W; j++) {
      int pos = (i * MAX_W + j) * 4;
      int inv = ((MAX_H - 1 - i) * MAX_W + j) * 4;
      for (int k = 0; k < image.metadata.channels; k++)
        ASSERT_EQ(image.data()[pos + k], ordered[inv + k]);
    }
}

TEST(ImageTest, flip_u) {
  std::unique_ptr<ImageHolderDataStorageInterface<int>> storage_data = std::make_unique<DataHostStorageTest<int>>();
  ImageHolderTest<int> image(std::move(storage_data));
  image.data().resize(MAX_H * MAX_W * 4, 0);
  image.metadata.channels = 4;
  image.metadata.height = MAX_H;
  image.metadata.width = MAX_W;

  for (int i = 0; i < MAX_H * MAX_W * 4; i++)
    image.data()[i] = i;
  std::vector<int> ordered(image.data().begin(), image.data().end());
  image.flip_u();
  for (int i = 0; i < MAX_H; i++)
    for (int j = 0; j < MAX_W; j++) {
      int pos = (i * MAX_W + j) * 4;
      int inv = (i * MAX_W + (MAX_W - 1 - j)) * 4;
      for (int k = 0; k < image.metadata.channels; k++)
        ASSERT_EQ(image.data()[pos + k], ordered[inv + k]);
    }
}
