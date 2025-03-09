#include "NovaTextureInterface.h"
namespace nova::texturing {

  glm::vec4 NovaTextureInterface::sample(float u, float v, const texture_sample_data &sample_data) const {
    auto disp = [&](auto texture) { return texture->sample(u, v, sample_data); };
    return dispatch(disp);
  }

  ConstantTexture::ConstantTexture(const glm::vec4 &albedo_) : albedo(albedo_) {}

  glm::vec4 ConstantTexture::sample(float /*u*/, float /*v*/, const texture_sample_data & /*sample_data*/) const { return albedo; }

  ImageTexture::ImageTexture(const uint32_t *raw_image, int w, int h, int chans, bool isrgba) : width(w), height(h), channels(chans) {
    AX_ASSERT_NOTNULL(raw_image);
    image_buffer = raw_image;
    is_rgba = isrgba;
  }

  /* Should be replaced with a solution that handles out of range UV mapping. */
  ax_device_callable_inlined unsigned uv2index(float t, int dim) {
    float a = AX_GPU_ABS(t);
    float rem = a - AX_GPU_FLOORF(a);
    unsigned i = math::texture::uvToPixel(rem, dim - 1);
    return i;
  }

  glm::vec4 ImageTexture::sample(float u, float v, const texture_sample_data & /*sample_data*/) const {
    unsigned i = uv2index(u, width);
    unsigned j = uv2index(v, height);
    unsigned idx = (i * height + j);
    AX_ASSERT_LT(idx, width * height);

    union FORMAT {
      uint32_t rgba;
      struct {
        uint8_t r;
        uint8_t g;
        uint8_t b;
        uint8_t a;
      };
    };
    FORMAT pixel{};
    pixel.rgba = image_buffer[idx];
    uint8_t r;
    uint8_t g;
    uint8_t b;
    uint8_t a;

    if (is_rgba) {
      r = pixel.r;
      g = pixel.g;
      b = pixel.b;
      a = pixel.a;
    } else {
      r = pixel.b;
      g = pixel.g;
      b = pixel.r;
      a = pixel.a;
    }

    switch (channels) {
      case 1:
        return {r, 0, 0, 1.f};
      case 2:
        return {r, g, 0, 1.f};
      case 3:
        return {r, g, b, 1.f};
      case 4:
        return {r, g, b, a};
      default:
        AX_UNREACHABLE
        break;
    }
    AX_UNREACHABLE
    return glm::vec4(0);
  }

}  // namespace nova::texturing
