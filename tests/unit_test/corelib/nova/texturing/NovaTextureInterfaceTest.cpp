#include "Image.h"
#include "ImageImporter.h"
#include "Loader.h"
#include "Metadata.h"
#include "Test.h"
#include "texturing/TextureContext.h"
#include "texturing/texture_datastructures.h"
#include <gtest/gtest.h>
#include <internal/common/math/math_spherical.h>
#include <internal/common/math/math_texturing.h>

#include "texturing/NovaTextureInterface.h"

namespace nvt = nova::texturing;
class TextureCtxFake : public nvt::TextureCtx {
 public:
  TextureCtxFake(const nvt::TextureBundleViews &bundle) : TextureCtx(bundle) {}
};

class SharedFloatTexturesStub {
  std::vector<float> f32_texture;

 public:
  constexpr static std::size_t SIZE = 256 * 256 * 3;
  constexpr static float RED = 0.4f;
  constexpr static float GREEN = 0.5f;
  constexpr static float BLUE = 0.6f;

  SharedFloatTexturesStub() {
    f32_texture.resize(SIZE);
    for (int i = 0; i < SIZE; i += 3) {
      f32_texture[i] = RED;
      f32_texture[i + 1] = GREEN;
      f32_texture[i + 2] = BLUE;
    }
  }

  F32Texture getF32() const {
    F32Texture tex;
    tex.width = 256;
    tex.height = 256;
    tex.channels = 3;
    tex.is_rgba = false;
    tex.raw_data = f32_texture;
    return tex;
  }
};

static nvt::TextureCtx create_aggregate(const std::array<F32Texture, 1> &array) {

  nvt::f32tex_shared_views_s shared_views;
  shared_views.f32_managed = array;
  nvt::TextureBundleViews bundle(shared_views);
  nvt::TextureCtx ctx(bundle);
  return ctx;
}

TEST(EnvmapTextureTest, sample_RGBA_order) {
  SharedFloatTexturesStub stub;

  std::array<F32Texture, 1> array = {stub.getF32()};
  nvt::TextureCtx ctx = create_aggregate(array);
  nvt::texture_data_aggregate_s aggregate;
  aggregate.texture_ctx = &ctx;
  aggregate.geometric_data.sampling_vector = {0.f, 1.f, 0.f};

  nvt::EnvmapTexture envmap(0);
  glm::vec4 pixel = envmap.sample(0.f, 0.f, aggregate);
  ASSERT_EQ(pixel.r, SharedFloatTexturesStub::RED);
  ASSERT_EQ(pixel.g, SharedFloatTexturesStub::GREEN);
  ASSERT_EQ(pixel.b, SharedFloatTexturesStub::BLUE);
}

class FTextureBuilderStub {
  std::vector<float> texture;

  void putColor(float r, float g, float b, std::size_t idx) {
    texture[idx] = r;
    texture[idx + 1] = g;
    texture[idx + 2] = b;
  }

 public:
  constexpr static std::size_t SIZE = 256 * 256 * 3;
  constexpr static std::size_t WIDTH = 256;
  constexpr static std::size_t HEIGHT = 256;

  // Encodes cubemap sampling direction as color.
  FTextureBuilderStub() {
    texture.resize(SIZE);
    for (int j = 0; j < HEIGHT; j++)
      for (int i = 0; i < WIDTH; i++) {
        std::size_t idx = (j * WIDTH + i);
        double u = math::texture::pixelToUv(i, WIDTH - 1);
        double v = math::texture::pixelToUv(j, HEIGHT - 1);
        glm::vec2 sph = math::spherical::uvToSpherical({u, v});
        glm::vec3 cart = math::spherical::sphericalToCartesian(sph);
        putColor(cart.r, cart.g, cart.b, idx);
      }

    image::Metadata metadata;
    metadata.channels = 3;
    metadata.width = WIDTH;
    metadata.height = HEIGHT;
    metadata.name = "envmaptest";
    metadata.format = "hdr";
    metadata.color_corrected = true;
    metadata.is_hdr = true;
    image::ImageHolder<float> image(texture, metadata);
    IO::Loader loader(nullptr);
    loader.writeHdr("/tmp/envmaptest", image);
  }

  F32Texture getF32() const {
    F32Texture tex;
    tex.width = WIDTH;
    tex.height = HEIGHT;
    tex.channels = 3;
    tex.raw_data = texture;
    return tex;
  }
};

TEST(EnvmapTextureTest, sample) {
  FTextureBuilderStub stub;
  std::array<F32Texture, 1> array = {stub.getF32()};
  nvt::TextureCtx ctx = create_aggregate(array);
  nvt::texture_data_aggregate_s aggregate;
  aggregate.texture_ctx = &ctx;

  nvt::EnvmapTexture envmap(0);
  glm::vec3 sample_vector = {0.f, 1.f, 0.f};
  aggregate.geometric_data.sampling_vector = sample_vector;
  glm::vec4 pixel = envmap.sample(0, 0, aggregate);
  ASSERT_EQ(pixel.r, sample_vector.r);
  ASSERT_EQ(pixel.g, sample_vector.g);
  ASSERT_EQ(pixel.b, sample_vector.b);
}
