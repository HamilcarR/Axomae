#include "SDL\SDL_surface.h"
#include <cuda.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <cstdint>
#include <stdio.h>
#include <iostream>
#include <fstream>

inline void write_file_arrays(void* host_array, int size_w, int size_h , int bpp , int pitch , std::string T) {
	std::ofstream file; 
	file.open(T.c_str() , std::ios::out | std::ios::ate | std::ios::trunc); 
	for (int i = 0; i < size_w; i++) {
		for (int j = 0; j < size_h; j++) {
			file << " host : " << (int)((uint8_t*)host_array)[i*bpp + j*pitch] << "\n";
		}
	}
	file.close(); 
}




 uint32_t* GPU_Initialize(int w, int h); 




 void GPU_compute_greyscale(SDL_Surface* image,  const bool bigEndian);