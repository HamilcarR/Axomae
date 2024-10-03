#ifndef MATH_TEXTURING_H
#define MATH_TEXTURING_H
#include "internal/device/gpgpu/device_utils.h"
namespace math::texture {
  constexpr float ONE_255 = 1.f / 255.f;

  template<class T>
  AX_DEVICE_CALLABLE float rgb_uint2float(T rgb) {
    return (float)rgb * ONE_255;
  }

  template<class D>
  AX_DEVICE_CALLABLE double pixelToUv(D coord, const unsigned dim) {
    return static_cast<double>(coord) / static_cast<double>(dim);
  }

  template<class D>
  AX_DEVICE_CALLABLE unsigned uvToPixel(D coord, unsigned dim) {
    return static_cast<unsigned>(coord * dim);
  }
}  // namespace math::texture
#endif  // math_texturing_H
