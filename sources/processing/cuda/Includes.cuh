#ifndef INCLUDES_CUH
#define INCLUDES_CUH
#include "Logger.h"
#include "constants.h"
#include "cuda_utils.h"
#include "utils_3D.h"

#include <texture_types.h>

constexpr glm::dvec3 UP_VECTOR = glm::dvec3(0.f, 1.f, 0.f);

AX_DEVICE_CALLABLE inline float magnitude(float x, float y) { return sqrtf(x * x + y * y); }

template<typename U, typename T>
AX_DEVICE_CALLABLE inline T normalize(U maxx, U minn, T pixel) {
  assert(maxx - minn != 0);
  return ((pixel - minn) * 255 / (maxx - minn) + 0);
}

template<typename T, typename D>
AX_DEVICE_CALLABLE auto lerp(T value1, T value2, D cste) {
  return (1 - cste) * value1 + cste * value2;
}

struct gpu_threads {
  dim3 threads;
  dim3 blocks;
};

inline void check_error(const char *file, int line) {
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::string err_str = "CUDA ERROR : " + std::string(cudaGetErrorString(err));
    LOG(err_str, LogLevel::ERROR);
  }
}

/**
 * @brief Get the maximum blocks/threads for a 2D buffer
 */
inline gpu_threads get_optimal_thread_distribution(float width, float height, float depth = 0) {
  gpu_threads value;
  float flat_array_size = width * height;
  /*need compute capability > 2.0*/
  dim3 threads = dim3(32, 32, 1);
  value.threads = threads;
  if (flat_array_size <= static_cast<float>(threads.y * threads.x)) {
    dim3 blocks = dim3(1);
    value.blocks = blocks;
  } else {
    float divx = width / threads.x;
    float divy = height / threads.y;
    float divz = depth / threads.x;
    int blockx = std::floor(divx) + 1;
    int blocky = std::floor(divy) + 1;
    int blockz = std::floor(divz) + 1;
    dim3 blocks(blockx, blocky, blockz);
    value.blocks = blocks;
  }
  return value;
}

#endif