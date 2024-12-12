#ifndef SAMPLER_H
#define SAMPLER_H
#include "internal/common/math/math_random.h"
#include "internal/common/math/math_utils.h"
#include "internal/device/gpgpu/device_utils.h"
#include "internal/macro/project_macros.h"
#include "internal/memory/tag_ptr.h"

#include <engine/nova_exception.h>
#include <memory>
#include <variant>
#if defined(AXOMAE_USE_CUDA)
#  include "gpu/GPURandomGenerator.h"
#endif

namespace nova::sampler {

#ifndef __CUDA_ARCH__
  class RandomSampler {
    exception::NovaException exception{};
    math::random::CPUPseudoRandomGenerator generator;

   public:
    CLASS_DCM(RandomSampler)

    ax_host_only explicit RandomSampler(uint64_t seed) : generator(seed) {}
    ax_host_only glm::vec3 sample() { return {generator.nrandf(-1, 1), generator.nrandf(-1, 1), generator.nrandf(-1, 1)}; }
    ax_host_only ax_no_discard exception::NovaException getErrorState() const { return exception; }
  };

  class SobolSampler {
    math::random::CPUQuasiRandomGenerator generator;
    exception::NovaException exception{};

   public:
    CLASS_DCM(SobolSampler)

    ax_host_only explicit SobolSampler(int seed, int dimension = 1) {
      if (seed <= 0 || dimension <= 0)
        exception.addErrorType(nova::exception::INVALID_SAMPLER_DIM);
      else {
        generator = math::random::CPUQuasiRandomGenerator(seed, dimension);
      }
    }
    ax_host_only ax_no_discard glm::vec3 sample() { return {generator.nrandf(-1, 1), generator.nrandf(-1, 1), generator.nrandf(-1, 1)}; }
    ax_host_only ax_no_discard exception::NovaException getErrorState() const { return exception; }
  };
#else
  class RandomSampler {
    exception::NovaException exception{};
    math::random::GPUPseudoRandomGenerator generator;

   public:
    CLASS_DCM(RandomSampler)
    /* Maybe for a HIP compatible implementation*/
    template<class T>
    ax_device_only RandomSampler(T *gpu_rand_states_buffer, uint64_t seed) : generator(gpu_rand_states_buffer, seed) {}
    ax_device_only explicit RandomSampler(const math::random::GPUPseudoRandomGenerator &gen) { generator = gen; }
    ax_device_only glm::vec3 sample() { return {generator.nrandf(-1, 1), generator.nrandf(-1, 1), generator.nrandf(-1, 1)}; }
    ax_device_only ax_no_discard exception::NovaException getErrorState() const { return exception; }
  };

  class SobolSampler {
    exception::NovaException exception{};
    math::random::GPUQuasiRandomGenerator generator;

   public:
    CLASS_DCM(SobolSampler)

    template<class T>
    ax_device_only SobolSampler(T *gpu_rand_states_buffer, unsigned dimension) : generator(gpu_rand_states_buffer, dimension) {}
    ax_device_only SobolSampler(const math::random::GPUQuasiRandomGenerator &gen) { generator = gen; }
    ax_device_only ax_no_discard glm::vec3 sample() {
      // return {generator.nrandf(-1, 1), generator.nrandf(-1, 1), generator.nrandf(-1, 1)};
      return generator.nrand3f(-1, 1);
    }
    ax_device_only ax_no_discard exception::NovaException getErrorState() const { return exception; }
  };
#endif

  class SamplerInterface : public core::tag_ptr<SobolSampler, RandomSampler> {
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
