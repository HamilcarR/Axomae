#include "Sampler.h"
#include "math_importance_sampling.h"

GPU_CALLABLE glm::vec3 Sampler::pgc3d(unsigned x, unsigned y, unsigned z) { return math::importance_sampling::pgc3d(x, y, z); }
GPU_CALLABLE glm::vec3 Sampler::hammersley3D(unsigned i , unsigned N){
  return math::importance_sampling::hammersley3D(i , N);
}
