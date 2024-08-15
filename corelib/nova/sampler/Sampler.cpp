#include "Sampler.h"
#include "engine/nova_exception.h"
#include "math_importance_sampling.h"
#include <glm/gtc/matrix_access.hpp>

#include <boost/random/uniform_01.hpp>
using namespace nova::sampler;

AX_DEVICE_CALLABLE glm::vec3 RandomSampler::sample() {
  return {math::random::nrandi(-1, 1), math::random::nrandi(-1, 1), math::random::nrandi(-1, 1)};
}
nova::exception::NovaException RandomSampler::getErrorState() const { return exception; }

HammersleySampler::HammersleySampler(int N_) : N(N_) {}

AX_DEVICE_CALLABLE glm::vec3 HammersleySampler::sample() {
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
  try {
    sobol_engine = std::make_unique<boost::random::sobol>(dimension);
    sobol_engine->seed(seed);
  } catch (const std::range_error &e) {
    exception.addErrorType(nova::exception::GENERAL_ERROR);
    exception.addErrorType(nova::exception::SAMPLER_DOMAIN_EXHAUSTED);
  } catch (const std::bad_alloc &e) {
    exception.addErrorType(nova::exception::GENERAL_ERROR);
    exception.addErrorType(nova::exception::SAMPLER_INVALID_ALLOC);
  } catch (const std::invalid_argument &e) {
    exception.addErrorType(nova::exception::GENERAL_ERROR);
    exception.addErrorType(nova::exception::SAMPLER_INVALID_ARG);
  }
}

AX_DEVICE_CALLABLE glm::vec3 SobolSampler::sample() {

  boost::random::uniform_01<> dist;
  float x, y, z;
  glm::vec3 p;
  x = dist(*sobol_engine) * 2.f - 1.f;
  y = dist(*sobol_engine) * 2.f - 1.f;
  z = dist(*sobol_engine) * 2.f - 1.f;
  p = glm::vec3(x, y, z);

  return p;
}
