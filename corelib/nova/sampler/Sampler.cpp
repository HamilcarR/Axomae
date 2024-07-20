#include "Sampler.h"
#include "engine/nova_exception.h"
#include "math_importance_sampling.h"
#include <glm/gtc/matrix_access.hpp>

using namespace nova::sampler;
HammersleySampler::HammersleySampler(int N_) : N(N_) {}

GPU_CALLABLE glm::vec3 HammersleySampler::sample(int &error_flag) {
  if (N <= 0) {
    error_flag |= nova::exception::INVALID_SIZE;
    return glm::vec3(0.f);
  }
  i++;
  return math::importance_sampling::hammersley3D(i, N);
}

SobolSampler::SobolSampler(int N_, int dimension) : sobol_engine(dimension), N(N_) { sobol_engine.seed(N_); }

GPU_CALLABLE glm::vec3 SobolSampler::sample(int &error_flag) {
  float x, y, z;
  glm::vec3 p;
  x = dist(sobol_engine) * 2.f - 1.f;
  y = dist(sobol_engine) * 2.f - 1.f;
  z = dist(sobol_engine) * 2.f - 1.f;
  p = glm::vec3(x, y, z);

  return p;
}
