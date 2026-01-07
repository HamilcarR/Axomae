#ifndef MATH_TEXTURING_H
#define MATH_TEXTURING_H
#include <internal/device/gpgpu/device_macros.h>
#include <internal/device/gpgpu/device_utils.h>
namespace math::texture {
  constexpr float ONE_255 = 1.f / 255.f;

  template<class T>
  ax_device_callable_inlined float rgb_uint2float(T rgb) {
    return (float)rgb * ONE_255;
  }

  template<class D>
  ax_device_callable_inlined double pixelToUv(D coord, const unsigned dim) {
    return static_cast<double>(coord) / static_cast<double>(dim);
  }

  template<class D>
  ax_device_callable_inlined unsigned uvToPixel(D coord, unsigned dim) {
    return static_cast<unsigned>(coord * dim);
  }
}  // namespace math::texture
#endif  // math_texturing_H
