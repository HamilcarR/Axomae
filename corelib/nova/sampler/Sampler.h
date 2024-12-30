#ifndef SAMPLER_H
#define SAMPLER_H
#include "internal/common/math/math_random.h"
#include "internal/common/math/math_utils.h"
#include "internal/device/gpgpu/device_utils.h"
#include "internal/macro/project_macros.h"
#include "internal/memory/tag_ptr.h"

#include <engine/nova_exception.h>
#include <memory>

#if defined(AXOMAE_USE_CUDA)
#  include "gpu/GPURandomGenerator.h"
#endif

namespace nova::sampler {

  using PRandomGenerator = math::random::CPUPseudoRandomGenerator;
  using QRandomGenerator = math::random::CPUQuasiRandomGenerator;
#if defined AXOMAE_USE_CUDA
  using DPRandomGenerator = math::random::GPUPseudoRandomGenerator;
  using DQRandomGenerator = math::random::GPUQuasiRandomGenerator;
#else
  using DPRandomGenerator = PRandomGenerator;
  using DQRandomGenerator = QRandomGenerator;
#endif

  template<class T>
  class SobolSampler {
   private:
    exception::NovaException exception{};
    T generator;

   public:
    CLASS_DCM(SobolSampler)

    ax_device_callable explicit SobolSampler(const T &generator_) : generator(generator_) {}
    ax_device_callable ax_no_discard glm::vec3 sample() { return generator.nrand3f(-1, 1); }
    template<class U>
    ax_device_callable ax_no_discard glm::vec3 sample(U min, U max) {
      return generator.nrand3f(min, max);
    }
    ax_device_callable ax_no_discard exception::NovaException getErrorState() const { return exception; }
  };

  template<class T>
  class RandomSampler {
  private:
    exception::NovaException exception{};
    T generator;

   public:
    CLASS_DCM(RandomSampler)

    ax_device_callable explicit RandomSampler(const T &generator_) : generator(generator_) {}
    ax_device_callable ax_no_discard glm::vec3 sample() { return generator.nrand3f(-1, 1); }
    template<class U>
    ax_device_callable ax_no_discard glm::vec3 sample(U min, U max) {
      return generator.nrand3f(min, max);
    }
    ax_device_callable ax_no_discard exception::NovaException getErrorState() const { return exception; }
  };

  class SamplerInterface : public core::tag_ptr<SobolSampler<QRandomGenerator>,
                                                SobolSampler<DQRandomGenerator>,
                                                RandomSampler<PRandomGenerator>,
                                                RandomSampler<DPRandomGenerator>> {
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
    template<class T>
    ax_device_callable ax_no_discard glm::vec3 sample(T min, T max) {
      auto d = [&](auto ptr) { return ptr->sample(min, max); };
      return dispatch(d);
    }
  };

  inline exception::NovaException retrieve_sampler_error(const SamplerInterface &sampler) { return sampler.getErrorState(); }
}  // namespace nova::sampler
#endif  // SAMPLER_H
