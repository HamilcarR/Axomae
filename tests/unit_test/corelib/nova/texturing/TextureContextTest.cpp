#include "texturing/TextureContext.h"
#include "Test.h"
#include "texturing/texture_datastructures.h"
#include <gtest/gtest.h>
#include <internal/macro/project_macros.h>

namespace nvt = nova::texturing;

template<class T>
struct type_texview_converter_s;

template<>
struct type_texview_converter_s<float> {
  using view_type = nvt::f32tex_shared_views_s;
  using return_type = glm::vec4;
  // clang-format off
  static constexpr float buffer[] = 
  {
    1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f ,
    9.f , 10.f , 11.f , 12.f , 13.f , 14.f , 15.f , 16.f
  };
  // clang-format on
  static constexpr unsigned width = 2;
  static constexpr unsigned height = 2;
  static constexpr unsigned channels = 4;
  static constexpr bool is_rgba = false;
  static constexpr bool is_u_inverted = false;
  static constexpr bool is_v_inverted = false;

  static constexpr unsigned size = sizeof(buffer) / sizeof(float);
  static return_type at(unsigned idx) {
    return {buffer[idx * channels], buffer[idx * channels + 1], buffer[idx * channels + 2], buffer[idx * channels + 3]};
  }
};

template<>
struct type_texview_converter_s<uint32_t> {
  using view_type = nvt::u32tex_shared_views_s;
  using return_type = uint32_t;
  static constexpr uint32_t buffer[] = {0xFF000000, 0x00FF0000, 0x0000FF00, 0x000000FF};
  static constexpr unsigned width = 2;
  static constexpr unsigned height = 2;
  static constexpr unsigned channels = 4;
  static constexpr bool is_rgba = false;
  static constexpr bool is_u_inverted = false;
  static constexpr bool is_v_inverted = false;
  static constexpr unsigned size = sizeof(buffer) / sizeof(uint32_t);
  static return_type at(unsigned idx) { return buffer[idx]; }
};

template<class T, class U = type_texview_converter_s<T>>
class SharedViewBuilder {
  using tex_views_t = typename U::view_type;
  using return_type = typename U::return_type;
  std::vector<nvt::TextureRawData<T>> textures;

 public:
  SharedViewBuilder() {
    nvt::TextureRawData<T> tex;
    tex.width = U::width;
    tex.height = U::height;
    tex.channels = U::channels;
    tex.is_rgba = U::is_rgba;
    tex.invert_u = U::is_u_inverted;
    tex.invert_v = U::is_v_inverted;
    tex.raw_data = axstd::span(U::buffer, U::size);
    textures.push_back(tex);
  }

  return_type valueAt(unsigned idx) const { return U::at(idx); }

  tex_views_t getTexture() const {
    tex_views_t shared_view;
    axstd::span<const nvt::TextureRawData<T>> texture_view(textures);
    shared_view.managed_tex_view = texture_view;
    return shared_view;
  }
};

void test_u32_tex() {
  SharedViewBuilder<uint32_t> builder;
  nvt::TextureBundleViews tbv(builder.getTexture());
  nvt::TextureCtx ctx(tbv);
  uint32_t pixel = ctx.u32pixel(0, 0.f, 0.f);
  ASSERT_EQ(pixel, builder.valueAt(0));
  pixel = ctx.u32pixel(0, 1.f, 0.f);
  ASSERT_EQ(pixel, builder.valueAt(1));
  pixel = ctx.u32pixel(0, 0.f, 1.f);
  ASSERT_EQ(pixel, builder.valueAt(2));
  pixel = ctx.u32pixel(0, 1.f, 1.f);
  ASSERT_EQ(pixel, builder.valueAt(3));
}

void test_f32_tex() {
  SharedViewBuilder<float> builder;
  nvt::TextureBundleViews tbv(builder.getTexture());
  nvt::TextureCtx ctx(tbv);
  glm::vec4 pixel = ctx.f32pixel(0, 0.f, 0.f);
  ASSERT_EQ(pixel, builder.valueAt(0));
  pixel = ctx.f32pixel(0, 1.f, 0.f);
  ASSERT_EQ(pixel, builder.valueAt(1));
  pixel = ctx.f32pixel(0, 0.f, 1.f);
  ASSERT_EQ(pixel, builder.valueAt(2));
  pixel = ctx.f32pixel(0, 1.f, 1.f);
  ASSERT_EQ(pixel, builder.valueAt(3));
}

TEST(TextureContextTest, sample_texture) {
  test_u32_tex();
  test_f32_tex();
}
