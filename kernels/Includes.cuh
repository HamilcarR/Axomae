#ifndef INCLUDES_CUH
#define INCLUDES_CUH
#include <cuda.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>
#include <texture_fetch_functions.h>
#include <texture_types.h>
#include <device_launch_parameters.h>
#include "../includes/constants.h"



#define cudaErrCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
        if (code != cudaSuccess) 
        {
                fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
                if (abort) exit(code);
        }
}




constexpr double PI = 3.14159265358979323846264; 
constexpr glm::dvec3 UP_VECTOR = glm::dvec3(0.f , 1.f , 0.f);



__host__ __device__
inline float magnitude(float x, float y) {return sqrtf(x*x + y*y);}


template<typename U , typename  T> 
__host__ __device__
inline T normalize(U maxx, U minn, T pixel) {
	assert(maxx - minn != 0);
	return ((pixel - minn) * 255 / (maxx - minn) + 0);
}

template<typename T, typename D> 
__host__ __device__
auto lerp(T value1, T value2, D cste) {
	return (1 - cste) * value1 + cste * value2;
}

struct gpu_threads {
	dim3 threads;
	dim3 blocks;
};

inline void check_error(const char* file , int line) {
	cudaError_t err = cudaGetLastError(); 
	if (err != cudaSuccess) {
		std::cout << "CUDA ERROR at file :  " << file << " Line : " << line << " => " << cudaGetErrorString(err) << "\n"; 
	}
}

/**
 * @brief Get the maximum blocks/threads for a 2D buffer
 * 
 * @param width Width of the buffer
 * @param height Height of the buffer
 * @return gpu_threads 
 */
inline gpu_threads get_optimal_thread_distribution(float width, float height , float depth = 0) {
	gpu_threads value;
	float flat_array_size = width*height;
	/*need compute capability > 2.0*/
	dim3 threads = dim3(32, 32 , 1);
	value.threads = threads;
	if (flat_array_size <= static_cast<float>(threads.y * threads.x)) {
		dim3 blocks = dim3(1);
		value.blocks = blocks;
	}
	else {
		float divx = width / threads.x;
		float divy = height / threads.y;
		float divz = depth / threads.x ; 
		int blockx =  std::floor(divx) + 1;
		int blocky =  std::floor(divy) + 1;
		int blockz =  std::floor(divz) + 1;
		dim3 blocks(blockx, blocky , blockz);
		value.blocks = blocks;
	}
	return value;
}








#endif