#include <cuda.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <cstdint>
#include <stdio.h>

__global__ void  GPU_compute_hmap(uint32_t* array, int width, int height);
__global__ void  GPU_compute_greyscale_luminance(uint32_t *array, int width, int height); 
__global__ void GPU_compute_nmap();
__global__ void GPU_compute_greyscale();
__global__ void GPU_set_greyscale();






 uint32_t* GPU_Initialize(int w, int h); 
 uint32_t* GPU_compute_greyscale(uint32_t* array, int width, int height); 