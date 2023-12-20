#ifndef KERNEL_CUH
#define KERNEL_CUH
#include "Includes.cuh"
#include <SDL2/SDL_surface.h>
#include <cstdint>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <assert.h>


namespace axomae{
/**we define some constants here...flags,sobel-prewitt operators,kernels etc...***************************************************************************/
	


__device__
static const float mean_smoothing_3_3[KERNEL_SIZE][KERNEL_SIZE] = {
	{1 , 1 , 1},
	{1 , 1 , 1},
	{1 , 1 , 1}
};

__device__
static const float sharpen_kernel[KERNEL_SIZE][KERNEL_SIZE] = {
	{0. , -1. , 0.} ,
	{-1. , 5. , -1.} , 
	{0. , -1. , 0.} 
};

__device__
static const float gaussian_blur_5_5[5][5] = {
	{1./256 , 4./256 , 6./256 , 4./256 , 1./256},
	{4./256 , 16./256 , 24./256 , 16./256 , 4./256},
	{6./256 , 24./256, 36./256 , 24./256 , 6./256}, 
	{4./256 , 16./256 , 24./256 , 16./256 , 4./256},
	{1./256 , 4./256 , 6./256 , 4./256 , 1./256} 
};


__device__
static const float unsharp_masking[5][5] = {
	{-1./256 ,-4./256 , -6./256 , -4./256 , -1./256},
	{-4./256 , -16./256 , -24./256 , -16./256 , -4./256},
	{-6./256 , -24./256, -476./256 , -24./256 , -6./256}, 
	{-4./256 , -16./256 , 24./256 , -16./256 , -4./256},
	{-1./256 , -4./256 , 6./256 , -4./256 , -1./256} 
};

__device__
static const float box_blur[KERNEL_SIZE][KERNEL_SIZE] = {
		{1./9 , 1./9 , 1./9},
		{1./9 , 1./9 , 1./9},
		{1./9 , 1./9 , 1./9}
};
__device__
static const float gaussian_blur_3_3[KERNEL_SIZE][KERNEL_SIZE] = {
		{1./16 , 2./16 , 1./16},
		{2./16 , 4./16 , 2./16},
		{1./16 , 2./16 , 1./16}
};

__device__
const int scharr_mask_vertical[KERNEL_SIZE][KERNEL_SIZE] = {
			{ 3 , 10 , 3 },
			{ 0 , 0  , 0 },
			{ -3, -10 ,-3 }
};

 __device__
const int scharr_mask_horizontal[KERNEL_SIZE][KERNEL_SIZE] = {
			{ 3 , 0 , -3 },
			{ 10 , 0 , -10 },
			{ 3 , 0 , -3 }
};

 __device__
const int sobel_mask_vertical[KERNEL_SIZE][KERNEL_SIZE] = {
			{ -1,-SOBEL,-1 },
			{ 0,  0  ,0 },
			{ 1, SOBEL , 1 }
};

 __device__
const int sobel_mask_horizontal[KERNEL_SIZE][KERNEL_SIZE] = {
			{ 1,0,-1 },
			{ SOBEL,0,-SOBEL },
			{ 1,0,-1 }
};

 __device__
const int prewitt_mask_vertical[KERNEL_SIZE][KERNEL_SIZE] = {
			{ 1,PREWITT,1 },
			{ 0,  0  ,0 },
			{ -1, -PREWITT , -1 }
};

 __device__
const int prewitt_mask_horizontal[KERNEL_SIZE][KERNEL_SIZE] = {
			{ -1,0,1 },
			{ -PREWITT,0,PREWITT },
			{ -1,0,1 }
};


/*convolution kernels*******************************/


struct max_colors {

	uint8_t  max_rgb[3];
	uint8_t  min_rgb[3];

	__device__
		void init() {
		for (int i = 0; i < 3; i++) {
			max_rgb[i] = 0; 
			min_rgb[i] = 255; 
		}
	}
	__device__
		void compare_max(uint8_t r, uint8_t g, uint8_t b) {
		max_rgb[0] = max_rgb[0] >= r ? max_rgb[0] : r;
		max_rgb[1] = max_rgb[1] >= g ? max_rgb[1] : g;
		max_rgb[2] = max_rgb[2] >= b ? max_rgb[2] : b;

	}
	__device__
		void compare_min(uint8_t r, uint8_t g, uint8_t b) {
		min_rgb[0] = min_rgb[0] < r ? min_rgb[0] : r;
		min_rgb[1] = min_rgb[1] < g ? min_rgb[1] : g;
		min_rgb[2] = min_rgb[2] < b ? min_rgb[2] : b;
	}


};

/**************************************************/
inline void write_file_arrays(void* host_array, int size_w, int size_h, int bpp, int pitch, std::string T) {
	std::ofstream file;
	file.open(T.c_str(), std::ios::out | std::ios::ate | std::ios::trunc);
	for (int i = 0; i < size_w; i++) {
		for (int j = 0; j < size_h; j++) {
			file << " host : " << (int)((uint8_t*)host_array)[i*bpp + j*pitch] << "\n";
		}
	}
	file.close();
}

void GPU_compute_greyscale(SDL_Surface* image, const bool luminance);

void GPU_compute_height(SDL_Surface* image, uint8_t convolution_flag, uint8_t border_behaviour);

void GPU_compute_normal(SDL_Surface* image, double factor, uint8_t border_behaviour); 


}

#endif
