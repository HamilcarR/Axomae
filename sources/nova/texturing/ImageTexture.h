#ifndef IMAGETEXTURE_H
#define IMAGETEXTURE_H
#include "ConstantTexture.h"
#include "math_utils.h"

namespace nova::texturing {
  class ImageTexture : public NovaTextureInterface {
   private:
    const uint32_t *image_buffer{};
    int width{}, height{}, channels{};

   public:
    CLASS_OCM(ImageTexture)

    ImageTexture(const uint32_t *raw_image, int w, int h, int chans) : width(w), height(h), channels(chans) {
      AX_ASSERT_NOTNULL(raw_image);
      image_buffer = raw_image;
    }

    [[nodiscard]] glm::vec4 sample(float u, float v, const texture_sample_data &sample_data) const override {
      unsigned i = math::texture::uvToPixel(u, width - 1);
      unsigned j = math::texture::uvToPixel(v, height - 1);
      unsigned idx = (i * height + j);
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
      float r = math::texture::rgb_uint2float(pixel.r);
      float g = math::texture::rgb_uint2float(pixel.g);
      float b = math::texture::rgb_uint2float(pixel.b);
      float a = (float)pixel.a;
      switch (channels) {
        case 1:
          return {};
        case 2:
          return {};
        case 3:
          return {};
        case 4:
          return {b, g, r, a};
        default:
          AX_UNREACHABLE
          break;
      }
      AX_UNREACHABLE
      return glm::vec4(0.f);
    }
  };

}  // namespace nova::texturing
#endif  // IMAGETEXTURE_H
