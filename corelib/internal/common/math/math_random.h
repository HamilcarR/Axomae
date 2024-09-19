#ifndef MATH_RANDOM_H
#define MATH_RANDOM_H
#include "math_random_interface.h"
#include "math_utils.h"
#include <cstdint>
#include <internal/common/axstd/managed_buffer.h>
#include <internal/device/gpgpu/DeviceError.h>
#include <internal/device/gpgpu/device_macros.h>
#include <internal/device/gpgpu/device_transfer_interface.h>
#include <internal/device/gpgpu/device_utils.h>
#include <memory>
#include <sobol_direction_vectors.h>

namespace math::random {

  class CPUPseudoRandomGenerator : public AbstractRandomGenerator<CPUPseudoRandomGenerator> {
    class Impl;
    std::shared_ptr<Impl> pimpl{};

   public:
    CPUPseudoRandomGenerator();
    explicit CPUPseudoRandomGenerator(uint64_t seed);
    int nrandi(int min, int max);
    float nrandf(float min, float max);
    glm::vec3 nrand3f(float min, float max);
    bool randb();
  };

  class SobolGenerator : public RQMC<SobolGenerator, float> {
   public:
    using sample_type = float;
    using index_type = uint32_t;

   private:
    unsigned dimension;
    uint64_t seed;

    ax_device_callable_inlined index_type dv(unsigned dimension, unsigned seq) const {
#ifdef __CUDA_ARCH__
      AX_ASSERT_FALSE(d_direction_vectors_view.empty());
      return d_direction_vectors_view[dimension * SEQUENCE_SIZE + seq];
#else
      return sobol_direction_vectors[dimension][seq];
#endif
    }

    static constexpr float ONE_UINT_MAX = 1.f / float(0xFFFFFFFF);

    ax_device_callable_inlined uint32_t reverse_bits(uint32_t x) const {
      x = (((x & 0xaaaaaaaa) >> 1) | ((x & 0x55555555) << 1));
      x = (((x & 0xcccccccc) >> 2) | ((x & 0x33333333) << 2));
      x = (((x & 0xf0f0f0f0) >> 4) | ((x & 0x0f0f0f0f) << 4));
      x = (((x & 0xff00ff00) >> 8) | ((x & 0x00ff00ff) << 8));
      return ((x >> 16) | (x << 16));
    }

    ax_device_callable_inlined uint32_t hash(uint32_t x, uint32_t seed) const {
      x ^= x * 0x3d20adea;
      x += seed;
      x *= (seed >> 16) | 0x1;
      x ^= x * 0x05526c56;
      x ^= x * 0x53a22864;
      return x;
    }

    ax_device_callable_inlined uint32_t owen_scramble(uint32_t x, uint32_t seed) const {
      x = reverse_bits(x);
      x = hash(x, seed);
      return reverse_bits(x);
    }

    ax_device_callable_inlined uint32_t gray_code(uint32_t i) const { return i ^ (i >> 1); }

    ax_device_callable_inlined float sobol_sample(uint32_t index, unsigned dim, uint32_t seed) const {
      AX_ASSERT_LT(dim, DIMENSIONS_SIZE);
      uint32_t accum = 0;
      uint32_t gc = gray_code(index);
      for (unsigned i = 0; i < SEQUENCE_SIZE; i++) {
        if ((gc >> i) & 0x1) {
          accum ^= dv(dim, i);
        }
      }
      accum = owen_scramble(uint32_t(accum), seed);
      float result = float(accum) * ONE_UINT_MAX;
      return result;
    }

   public:
    ax_device_callable SobolGenerator() : seed(0xDEADBEEF), dimension(DIMENSIONS_SIZE) {}

    ax_device_callable SobolGenerator(uint64_t s) : seed(s), dimension(DIMENSIONS_SIZE) {}

    ax_device_callable float generate(uint64_t index, unsigned dimension) const {
      AX_ASSERT_LT(index, std::numeric_limits<uint32_t>::max());
      float point = sobol_sample(uint32_t(index), dimension, seed);
      AX_ASSERT(!ISNAN(point), "Sobol bad sample generation.");
      return point;
    }

#ifdef AXOMAE_USE_CUDA
   private:
    axstd::span<uint32_t> d_direction_vectors_view;

   public:
    ax_host_only void allocDeviceLookupTable() {
      auto err = device::gpgpu::allocate_buffer(DIMENSIONS_SIZE * SEQUENCE_SIZE * sizeof(index_type));
      DEVICE_ERROR_CHECK(err.error_status);
      d_direction_vectors_view = axstd::span<uint32_t>(static_cast<uint32_t *>(err.device_ptr), DIMENSIONS_SIZE * SEQUENCE_SIZE);
      err = device::gpgpu::copy_buffer(
          sobol_direction_vectors, d_direction_vectors_view.data(), d_direction_vectors_view.size() * sizeof(uint32_t), device::gpgpu::HOST_DEVICE);
      DEVICE_ERROR_CHECK(err.error_status);
    }

    ax_host_only void deallocDeviceLookupTable() {
      auto err = device::gpgpu::deallocate_buffer(static_cast<void *>(d_direction_vectors_view.data()));
      DEVICE_ERROR_CHECK(err.error_status);
      d_direction_vectors_view = {};
    }

#endif
  };

};  // namespace math::random
#endif  // math_random_H
