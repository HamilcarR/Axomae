#ifndef IMAGES_UTILS_H
#define IMAGES_UTILS_H
#include "Axomae_macros.h"

namespace hdr_utils {

  /* Disable strict aliasing warning on compilation because those functions need type prunning */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
  inline void float2rgbe(unsigned char rgbe[4], float r, float g, float b) {
    float mantissa = NAN;
    int exponent = 0;
    mantissa = std::max(std::max(r, g), b);
    if (mantissa < 1e-32)
      std::memset(rgbe, 0, 4);
    else {
      mantissa = std::frexp(mantissa, &exponent) * 256 / mantissa;
      rgbe[0] = (unsigned char)(r * mantissa);
      rgbe[1] = (unsigned char)(g * mantissa);
      rgbe[2] = (unsigned char)(b * mantissa);
      rgbe[3] = (unsigned char)(mantissa + 128);
    }
  }

  inline void rgbe2float(float rgbe, float rgb[3]) {
    float f = 0.f;
    uint8_t exp = *((uint32_t *)&rgbe) << 24;
    uint8_t b = *((uint32_t *)&rgbe) << 16;
    uint8_t g = *((uint32_t *)&rgbe) << 8;
    uint8_t r = *((uint32_t *)&rgbe) & 0xFF;
    f = std::ldexp(1, exp - (int)(128 + 8));
    rgb[0] = r * f;
    rgb[1] = g * f;
    rgb[2] = b * f;
  }
#pragma GCC diagnostic pop

  static constexpr float _gamma = 2.2;
  static constexpr float _exposure = 1.f;
  template<class T>
  inline T tone_mapping_channel(T channel) {
    return 1.f - std::exp(-channel * _exposure);
  }
  template<class T>
  inline T color_correction(T channel) {
    return std::pow(tone_mapping_channel(channel), 1.f / _gamma);
  }

  template<class T>
  inline T inv_tone_mapping_channel(T value) {
    return -std::log(1.f - value) / _exposure;
  }

  template<class T>
  inline T inv_color_correct(T value) {
    return std::exp(std::log(value) / (1.f / _gamma));
  }

  /* Creates a viewable image from an HDR image.
   * The output image is always of format RGBA for simplicity.
   * */
  template<class T>
  std::vector<uint8_t> hdr2image(const std::vector<T> &rgb, int width, int height, int channels, bool needs_color_correct) {
    ASSERT_IS_ARITHMETIC(T);
    AX_ASSERT(rgb.size() == static_cast<unsigned>(width * height * channels), "");

    std::vector<uint8_t> image_data;
    int final_channels = 4;
    image_data.resize(width * height * final_channels);

    /* Need max value for image to compute final pixel*/
    T max = 0;
    for (T i : rgb)
      max = std::max(max, i);
    AX_ASSERT(max > 0, "");
    if (needs_color_correct)
      max = color_correction(max);
    int index = 0;
    for (int i = 0; i < height; i++) {
      for (int j = 0; j < width; j++) {
        index = (i * width + j) * channels;
        T r, g, b;
        if (needs_color_correct) {
          r = (color_correction(rgb[index])) / max;
          g = (color_correction(rgb[index + 1])) / max;
          b = (color_correction(rgb[index + 2])) / max;
        } else {
          r = (rgb[index]) / max;
          g = (rgb[index + 1]) / max;
          b = (rgb[index + 2]) / max;
        }
        image_data[(i * width + j) * final_channels] = std::clamp(static_cast<int>(b * 255), 0, 255);
        image_data[(i * width + j) * final_channels + 1] = std::clamp(static_cast<int>(g * 255), 0, 255);
        image_data[(i * width + j) * final_channels + 2] = std::clamp(static_cast<int>(r * 255), 0, 255);
        image_data[(i * width + j) * final_channels + 3] = 0;
      }
    }
    return image_data;
  }

};  // namespace hdr_utils

#endif
