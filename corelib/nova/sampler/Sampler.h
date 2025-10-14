#ifndef SAMPLER_H
#define SAMPLER_H
#include "engine/nova_exception.h"
#include <internal/common/math/math_random.h>
#include <internal/common/math/math_utils.h>
#include <internal/device/gpgpu/device_macros.h>
#include <internal/device/gpgpu/device_utils.h>
#include <internal/macro/project_macros.h>
#include <internal/memory/tag_ptr.h>

#if defined(AXOMAE_USE_CUDA)
#  include <internal/common/math/gpu/math_random_gpu.h>
#endif

namespace nova::sampler {
  template<class T = math::random::SobolGenerator>
  class SobolSampler {
   public:
    using sample_type = typename T::sample_type;
    using index_type = typename T::index_type;

   private:
    T generator;
    index_type index{0};
    unsigned dimension_counter{0};

   public:
    ax_device_callable SobolSampler() = default;

    ax_device_callable explicit SobolSampler(const T &generator_) : generator(generator_) {}

    ax_device_callable_inlined sample_type sample1D() { return generator.generate(index, dimension_counter++); }

    ax_device_callable_inlined void sample2D(sample_type out_values[2]) {
      out_values[0] = sample1D();
      out_values[1] = sample1D();
    }

    ax_device_callable_inlined void sample3D(sample_type out_values[3]) {
      out_values[0] = sample1D();
      out_values[1] = sample1D();
      out_values[2] = sample1D();
    }

    ax_device_callable void reset(index_type idx) {
      dimension_counter = 0;
      index = idx;
    }

    ax_device_callable unsigned getDimCounter() const { return dimension_counter; }

    ax_device_callable index_type getIndexCounter() const { return index; }

    ax_device_callable void next1D() { dimension_counter++; }
  };

  class SamplerInterface : public core::tag_ptr<SobolSampler<math::random::SobolGenerator>> {
   public:
    using tag_ptr::tag_ptr;

    /* Always return a random number on [0 , 1) interval.*/
    ax_device_callable decltype(auto) sample1D() {
      auto d = [&](auto s) { return s->sample1D(); };
      return dispatch(d);
    }

    ax_device_callable void sample2D(float out_values[2]) {
      auto d = [&](auto s) { return s->sample2D(out_values); };
      return dispatch(d);
    }

    ax_device_callable void sample3D(float out_values[3]) {
      auto d = [&](auto s) { return s->sample3D(out_values); };
      return dispatch(d);
    }

    ax_device_callable void reset(std::size_t index) {
      auto d = [&](auto s) { return s->reset(index); };
      return dispatch(d);
    }

    ax_device_callable unsigned getDimCounter() const {
      auto d = [&](auto s) { return s->getDimCounter(); };
      return dispatch(d);
    }

    ax_device_callable std::size_t getIndexCounter() const {
      auto d = [&](auto s) { return s->getIndexCounter(); };
      return dispatch(d);
    }

    ax_device_callable void next1D() {
      auto d = [&](auto d) { d->next1D(); };
      dispatch(d);
    }
  };

  ax_device_callable_inlined nova::exception::NovaException retrieve_sampler_error(const SamplerInterface &sampler) {
    return nova::exception::NovaException();
  }

}  // namespace nova::sampler
#endif  // SAMPLER_H
