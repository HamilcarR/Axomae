#ifndef SAMPLER_H
#define SAMPLER_H
#include "cuda_utils.h"
#include "math_random.h"
#include "math_utils.h"
#include "project_macros.h"

#include <boost/random/sobol.hpp>
#include <boost/random/uniform_01.hpp>
#include <engine/nova_exception.h>
#include <memory>
#include <variant>

namespace nova::sampler {

  class HammersleySampler {
    int i{0}, N{0};
    mutable nova::exception::NovaException exception{};

   public:
    CLASS_M(HammersleySampler)

    explicit HammersleySampler(int N);
    /* Will increment the index*/
    GPU_CALLABLE glm::vec3 sample();
    nova::exception::NovaException getErrorState() const { return exception; }
  };

  class SobolSampler {
    std::unique_ptr<boost::random::sobol> sobol_engine;
    boost::random::uniform_01<> dist;
    int i{0}, N{0};
    mutable nova::exception::NovaException exception{};

   public:
    CLASS_M(SobolSampler)
    explicit SobolSampler(int seed, int dimension = 1);
    GPU_CALLABLE glm::vec3 sample();
    nova::exception::NovaException getErrorState() const { return exception; }
  };

  using SamplerInterface = std::variant<HammersleySampler, SobolSampler>;

  inline nova::exception::NovaException retrieve_sampler_error(const SamplerInterface &sampler) {
    return std::visit([](auto &&sampler) { return sampler.getErrorState(); }, sampler);
  }
}  // namespace nova::sampler
#endif  // SAMPLER_H
