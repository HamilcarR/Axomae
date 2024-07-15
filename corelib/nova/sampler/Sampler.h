#ifndef SAMPLER_H
#define SAMPLER_H
#include "cuda_utils.h"
#include "math_utils.h"
class Sampler {
 public:
  GPU_CALLABLE static glm::vec3 pgc3d(unsigned x, unsigned y, unsigned z);
  GPU_CALLABLE static glm::vec3 hammersley3D(unsigned i, unsigned N);
};

#endif  // SAMPLER_H
