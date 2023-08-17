#ifndef INCLUDES_CUH
#define INCLUDES_CUH
#include <cuda.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include "../includes/constants.h"

constexpr double PI = 3.14159265358979323846264; 
constexpr glm::dvec3 UP_VECTOR = glm::dvec3(0.f , 1.f , 0.f);



__host__ __device__
static float magnitude(float x, float y) {return sqrtf(x*x + y*y);}


template<typename U , typename  T> 
__host__ __device__
static T normalize(U maxx, U minn, T pixel) {
	assert(maxx - minn != 0);
	return ((pixel - minn) * 255 / (maxx - minn) + 0);
}

template<typename T, typename D> 
__host__ __device__
auto lerp(T value1, T value2, D cste) {
	return (1 - cste) * value1 + cste * value2;
}













#endif