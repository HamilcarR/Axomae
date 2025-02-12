#ifndef UTILS_H
#define UTILS_H

namespace core::build {

#ifdef AXOMAE_USE_CUDA
  constexpr bool is_gpu_build = true;
#else
  constexpr bool is_gpu_build = false;
#endif

#ifdef AXOMAE_USE_EMBREE
  constexpr bool is_embree_build = true;
#else
  constexpr bool is_embree_build = false;
#endif

}  // namespace core::build

#endif
