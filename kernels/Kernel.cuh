#include <cuda.h>
#include <SDL/SDL.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <cstdint>
#include <stdio.h>

__global__ void  GPU_compute_hmap(uint32_t* array, int width, int height);
template<typename T> __global__ void  GPU_compute_greyscale_luminance(T* array,uint32_t* new_array ,int width, int height, constexpr bool isbigEndian);
__global__ void GPU_compute_nmap();
__global__ void GPU_compute_greyscale();
__global__ void GPU_set_greyscale();






 uint32_t* GPU_Initialize(int w, int h); 
 template <typename T>
 uint32_t* GPU_compute_greyscale(T* image, int width, int height, int bpp, constexpr bool bigEndian, int pitch);