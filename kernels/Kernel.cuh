#include "SDL\SDL_surface.h"
#include <cuda.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <cstdint>
#include <stdio.h>








 uint32_t* GPU_Initialize(int w, int h); 




 void GPU_compute_greyscale(SDL_Surface* image,  const bool bigEndian);