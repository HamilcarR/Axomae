#include "Image.h"
#include "Test.h"
#include "math_utils.h"

constexpr int MAX_W = 1919;
constexpr int MAX_H = 1080;

TEST(ImageTest, flip_v) {
  image::ImageHolder<int> image;
  image.data.resize(MAX_H * MAX_W * 4, 0);
  image.metadata.channels = 4;
  image.metadata.height = MAX_H;
  image.metadata.width = MAX_W;

  for (int i = 0; i < MAX_H * MAX_W * 4; i++)
    image.data[i] = i;
  std::vector<int> ordered = image.data;
  image.flip_v();
  for (int i = 0; i < MAX_H; i++)
    for (int j = 0; j < MAX_W; j++) {
      int pos = (i * MAX_W + j) * 4;
      int inv = ((MAX_H - 1 - i) * MAX_W + j) * 4;
      for (int k = 0; k < image.metadata.channels; k++)
        EXPECT_EQ(image.data[pos + k], ordered[inv + k]);
    }
}

TEST(ImageTest, flip_u) {
  image::ImageHolder<int> image;
  image.data.resize(MAX_H * MAX_W * 4, 0);
  image.metadata.channels = 4;
  image.metadata.height = MAX_H;
  image.metadata.width = MAX_W;

  for (int i = 0; i < MAX_H * MAX_W * 4; i++)
    image.data[i] = i;
  std::vector<int> ordered = image.data;
  image.flip_u();
  for (int i = 0; i < MAX_H; i++)
    for (int j = 0; j < MAX_W; j++) {
      int pos = (i * MAX_W + j) * 4;
      int inv = (i * MAX_W + (MAX_W - 1 - j)) * 4;
      for (int k = 0; k < image.metadata.channels; k++)
        EXPECT_EQ(image.data[pos + k], ordered[inv + k]);
    }
}