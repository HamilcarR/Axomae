#ifndef SAMPLER_H
#define SAMPLER_H
#include "cuda_utils.h"
#include "math_random.h"
#include "math_utils.h"
#include "project_macros.h"
#include <boost/random/sobol.hpp>
#include <boost/random/uniform_01.hpp>
#include <variant>

namespace nova::sampler {

  class HammersleySampler {
    int i{0}, N{0};

   public:
    CLASS_CM(HammersleySampler)

    explicit HammersleySampler(int N);
    /* Will increment the index*/
    GPU_CALLABLE glm::vec3 sample(int &error_flag);
  };

  class SobolSampler {
    boost::random::sobol sobol_engine;
    boost::random::uniform_01<> dist;
    int i{0}, N{0};

   public:
    CLASS_CM(SobolSampler)
    explicit SobolSampler(int N, int dimension = 1);
    GPU_CALLABLE glm::vec3 sample(int &error_flag);
  };

  using SamplerInterface = std::variant<HammersleySampler, SobolSampler>;

}  // namespace nova::sampler
#endif  // SAMPLER_H
