#include "Sampler.h"
#include "engine/nova_exception.h"
#include "internal/common/math/math_importance_sampling.h"
#include <boost/random/uniform_01.hpp>
using namespace nova::sampler;

RandomSampler::RandomSampler(uint64_t seed) : generator(seed) {}

glm::vec3 RandomSampler::sample() { return {generator.nrandf(-1, 1), generator.nrandf(-1, 1), generator.nrandf(-1, 1)}; }
nova::exception::NovaException RandomSampler::getErrorState() const { return exception; }

HammersleySampler::HammersleySampler(int N_) : N(N_) {}

glm::vec3 HammersleySampler::sample() {
  if (N <= 0) {
    exception.addErrorType(nova::exception::INVALID_SAMPLER_DIM);
    return glm::vec3(0.f);
  }
  i++;
  return math::importance_sampling::hammersley3D(i, N);
}

SobolSampler::SobolSampler(int seed, int dimension) : N(seed) {
  if (seed <= 0 || dimension <= 0)
    exception.addErrorType(nova::exception::INVALID_SAMPLER_DIM);
  else {
    sobol_engine = boost::random::sobol(dimension);
    sobol_engine.seed(seed);
  }
}

glm::vec3 SobolSampler::sample() {
  boost::random::uniform_01<> dist;
  float x, y, z;
  glm::vec3 p;
  x = dist(sobol_engine) * 2.f - 1.f;
  y = dist(sobol_engine) * 2.f - 1.f;
  z = dist(sobol_engine) * 2.f - 1.f;
  p = glm::vec3(x, y, z);

  return p;
}
