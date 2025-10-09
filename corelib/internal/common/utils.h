#ifndef UTILS_H
#define UTILS_H

namespace core::build {

#ifdef AXOMAE_USE_CUDA
  constexpr bool is_cuda_build = true;
#else
  constexpr bool is_cuda_build = false;
#endif

#ifdef AXOMAE_USE_EMBREE
  constexpr bool is_embree_build = true;
#else
  constexpr bool is_embree_build = false;
#endif

#ifdef AXOMAE_USE_HIP
  constexpr bool is_hip_build = true;
#else
  constexpr bool is_hip_build = false;
#endif

#ifdef AXOMAE_USE_OPENCL
  constexpr bool is_opencl_build = true;
#else
  constexpr bool is_opencl_build = false;
#endif

#ifdef AXOMAE_USE_SPECTRUM
  constexpr bool is_spectrum_build = true;
#else
  constexpr bool is_spectrum_build = false;
#endif

  constexpr bool is_gpu_build = is_cuda_build || is_hip_build || is_opencl_build;

  static_assert(static_cast<int>(is_cuda_build) + static_cast<int>(is_hip_build) + static_cast<int>(is_opencl_build) <= 1,
                "Only one GPU API build is allowed.(Set one of AXOMAE_USE_CUDA , AXOMAE_USE_HIP , AXOMAE_USE_OPENCL to ON , and the rest to OFF .");

}  // namespace core::build

#endif
