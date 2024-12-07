#ifndef SAMPLER_H
#define SAMPLER_H
#include "internal/common/math/math_random.h"
#include "internal/common/math/math_utils.h"
#include "internal/device/gpgpu/device_utils.h"
#include "internal/macro/project_macros.h"
#include "internal/memory/tag_ptr.h"
#include <boost/random/sobol.hpp>
#include <engine/nova_exception.h>
#include <memory>
#include <variant>
#if defined(AXOMAE_USE_CUDA)
#  include "gpu/GPURandomGenerator.h"
#endif

namespace nova::sampler {

  class RandomSampler {
    exception::NovaException exception{};
#ifndef __CUDA_ARCH__
    math::random::CPURandomGenerator generator;
#else
    math::random::GPURandomGenerator generator;
#endif

   public:
    CLASS_DCM(RandomSampler)

    ax_device_callable explicit RandomSampler(uint64_t seed);
    ax_device_callable ax_no_discard glm::vec3 sample();
    ax_device_callable ax_no_discard exception::NovaException getErrorState() const;
  };

  class HammersleySampler {
    int i{0}, N{0};
    exception::NovaException exception{};

   public:
    CLASS_DCM(HammersleySampler)

    explicit HammersleySampler(int N);
    /* Will increment the index*/
    ax_device_callable ax_no_discard glm::vec3 sample();
    ax_device_callable ax_no_discard exception::NovaException getErrorState() const { return exception; }
  };

  class SobolSampler {
    boost::random::sobol sobol_engine{1};
    exception::NovaException exception{};
    int i{0}, N{0};

   public:
    CLASS_DCM(SobolSampler)

    ax_device_callable explicit SobolSampler(int seed, int dimension = 1);
    ax_device_callable glm::vec3 sample();
    ax_device_callable ax_no_discard exception::NovaException getErrorState() const { return exception; }
  };

  class SamplerInterface : public core::tag_ptr<HammersleySampler, SobolSampler, RandomSampler> {
   public:
    using tag_ptr::tag_ptr;

    ax_device_callable ax_no_discard exception::NovaException getErrorState() const {
      auto d = [&](auto ptr) { return ptr->getErrorState(); };
      return dispatch(d);
    }

    ax_device_callable ax_no_discard glm::vec3 sample() {
      auto d = [&](auto ptr) { return ptr->sample(); };
      return dispatch(d);
    }
  };

  inline exception::NovaException retrieve_sampler_error(const SamplerInterface &sampler) { return sampler.getErrorState(); }
}  // namespace nova::sampler
#endif  // SAMPLER_H
