#ifndef IMAGETEXTURE_H
#define IMAGETEXTURE_H
#include "ConstantTexture.h"
#include "math_utils.h"

namespace nova::texturing {
  class ImageTexture : public NovaTextureInterface {

   private:
    const uint32_t *image_buffer{};
    int width{}, height{}, channels{};
    bool is_rgba;  // or bgra
   public:
    CLASS_OCM(ImageTexture)

    ImageTexture(const uint32_t *raw_image, int w, int h, int chans, bool isrgba = false) : width(w), height(h), channels(chans) {
      AX_ASSERT_NOTNULL(raw_image);
      image_buffer = raw_image;
      is_rgba = isrgba;
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
  };

}  // namespace nova::texturing
#endif  // IMAGETEXTURE_H
