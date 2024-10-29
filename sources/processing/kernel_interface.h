#ifndef KERNEL_INTERFACE_CUH
#define KERNEL_INTERFACE_CUH
#include "constants.h"
#include "internal/device/gpgpu/device_utils.h"
#include <assert.h>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdio.h>

// Old code , tb refactored.
struct SDL_Surface;

ax_device_callable inline float magnitude(float x, float y) { return sqrtf(x * x + y * y); }

template<typename U, typename T>
ax_device_callable inline T normalize(U maxx, U minn, T pixel) {
  assert(maxx - minn != 0);
  return ((pixel - minn) * 255 / (maxx - minn) + 0);
}

template<typename T, typename D>
ax_device_callable auto lerp(T value1, T value2, D cste) {
  return (1 - cste) * value1 + cste * value2;
}
namespace axomae {
  /**we define some constants here...flags,sobel-prewitt operators,kernels.
   * etc...***************************************************************************/

  constexpr uint8_t KERNEL_SIZE = 3;
  // clang-format off
ax_device_const
const float mean_smoothing_3_3[KERNEL_SIZE][KERNEL_SIZE] = {
	{1 , 1 , 1},
	{1 , 1 , 1},
	{1 , 1 , 1}
};

ax_device_const
const float sharpen_kernel[KERNEL_SIZE][KERNEL_SIZE] = {
	{0. , -1. , 0.} ,
	{-1. , 5. , -1.} ,
	{0. , -1. , 0.}
};

ax_device_const
const float gaussian_blur_5_5[5][5] = {
	{1./256 , 4./256 , 6./256 , 4./256 , 1./256},
	{4./256 , 16./256 , 24./256 , 16./256 , 4./256},
	{6./256 , 24./256, 36./256 , 24./256 , 6./256},
	{4./256 , 16./256 , 24./256 , 16./256 , 4./256},
	{1./256 , 4./256 , 6./256 , 4./256 , 1./256}
};


ax_device_const
const float unsharp_masking[5][5] = {
	{-1./256 ,-4./256 , -6./256 , -4./256 , -1./256},
	{-4./256 , -16./256 , -24./256 , -16./256 , -4./256},
	{-6./256 , -24./256, -476./256 , -24./256 , -6./256},
	{-4./256 , -16./256 , 24./256 , -16./256 , -4./256},
	{-1./256 , -4./256 , 6./256 , -4./256 , -1./256}
};

ax_device_const
const float box_blur[KERNEL_SIZE][KERNEL_SIZE] = {
		{1./9 , 1./9 , 1./9},
		{1./9 , 1./9 , 1./9},
		{1./9 , 1./9 , 1./9}
};
ax_device_const
const float gaussian_blur_3_3[KERNEL_SIZE][KERNEL_SIZE] = {
		{1./16 , 2./16 , 1./16},
		{2./16 , 4./16 , 2./16},
		{1./16 , 2./16 , 1./16}
};

ax_device_const
const int scharr_mask_vertical[KERNEL_SIZE][KERNEL_SIZE] = {
			{ 3 , 10 , 3 },
			{ 0 , 0  , 0 },
			{ -3, -10 ,-3 }
};

 ax_device_const
const int scharr_mask_horizontal[KERNEL_SIZE][KERNEL_SIZE] = {
			{ 3 , 0 , -3 },
			{ 10 , 0 , -10 },
			{ 3 , 0 , -3 }
};

 ax_device_const
const int sobel_mask_vertical[KERNEL_SIZE][KERNEL_SIZE] = {
			{ -1,-SOBEL,-1 },
			{ 0,  0  ,0 },
			{ 1, SOBEL , 1 }
};

 ax_device_const
const int sobel_mask_horizontal[KERNEL_SIZE][KERNEL_SIZE] = {
			{ 1,0,-1 },
			{ SOBEL,0,-SOBEL },
			{ 1,0,-1 }
};

 ax_device_const
const int prewitt_mask_vertical[KERNEL_SIZE][KERNEL_SIZE] = {
			{ 1,PREWITT,1 },
			{ 0,  0  ,0 },
			{ -1, -PREWITT , -1 }
};

 ax_device_const
const int prewitt_mask_horizontal[KERNEL_SIZE][KERNEL_SIZE] = {
			{ -1,0,1 },
			{ -PREWITT,0,PREWITT },
			{ -1,0,1 }
};
  // clang-format on

  struct max_colors {

    uint8_t max_rgb[3];
    uint8_t min_rgb[3];

    ax_device_callable void init() {
      for (int i = 0; i < 3; i++) {
        max_rgb[i] = 0;
        min_rgb[i] = 255;
      }
    }
    ax_device_callable void compare_max(uint8_t r, uint8_t g, uint8_t b) {
      max_rgb[0] = max_rgb[0] >= r ? max_rgb[0] : r;
      max_rgb[1] = max_rgb[1] >= g ? max_rgb[1] : g;
      max_rgb[2] = max_rgb[2] >= b ? max_rgb[2] : b;
    }
    ax_device_callable void compare_min(uint8_t r, uint8_t g, uint8_t b) {
      min_rgb[0] = min_rgb[0] < r ? min_rgb[0] : r;
      min_rgb[1] = min_rgb[1] < g ? min_rgb[1] : g;
      min_rgb[2] = min_rgb[2] < b ? min_rgb[2] : b;
    }
  };

  void GPU_compute_greyscale(SDL_Surface *image, bool luminance);

  void GPU_compute_height(SDL_Surface *image, uint8_t convolution_flag, uint8_t border_behaviour);

  void GPU_compute_normal(SDL_Surface *image, double factor, uint8_t border_behaviour);

}  // namespace axomae

#endif
