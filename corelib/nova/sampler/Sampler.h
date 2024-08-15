#ifndef SAMPLER_H
#define SAMPLER_H
#include "cuda_utils.h"
#include "math_random.h"
#include "math_utils.h"
#include "project_macros.h"
#include "tag_ptr.h"

#include <boost/random/sobol.hpp>
#include <engine/nova_exception.h>
#include <memory>
#include <variant>

namespace nova::sampler {

  class RandomSampler {
    mutable nova::exception::NovaException exception{};

   public:
    CLASS_M(RandomSampler)

    AX_DEVICE_CALLABLE glm::vec3 sample();
    AX_DEVICE_CALLABLE nova::exception::NovaException getErrorState() const;
  };

  class HammersleySampler {
    int i{0}, N{0};
    mutable nova::exception::NovaException exception{};

   public:
    CLASS_M(HammersleySampler)

    explicit HammersleySampler(int N);
    /* Will increment the index*/
    AX_DEVICE_CALLABLE glm::vec3 sample();
    AX_DEVICE_CALLABLE nova::exception::NovaException getErrorState() const { return exception; }
  };

  class SobolSampler {
    std::unique_ptr<boost::random::sobol> sobol_engine;
    int i{0}, N{0};
    mutable nova::exception::NovaException exception{};

   public:
    explicit SobolSampler(int seed, int dimension = 1);
    AX_DEVICE_CALLABLE glm::vec3 sample();
    AX_DEVICE_CALLABLE nova::exception::NovaException getErrorState() const { return exception; }
  };

  class SamplerInterface : public core::tag_ptr<HammersleySampler, SobolSampler, RandomSampler> {
   public:
    using tag_ptr::tag_ptr;

    AX_DEVICE_CALLABLE [[nodiscard]] nova::exception::NovaException getErrorState() const {
      auto d = [&](auto ptr) { return ptr->getErrorState(); };
      return dispatch(d);
    }

    AX_DEVICE_CALLABLE [[nodiscard]] glm::vec3 sample() {
      auto d = [&](auto ptr) { return ptr->sample(); };
      return dispatch(d);
    }
  };

  inline nova::exception::NovaException retrieve_sampler_error(const SamplerInterface &sampler) { return sampler.getErrorState(); }
}  // namespace nova::sampler
#endif  // SAMPLER_H
