#include "math_random.h"

namespace math::random {

  CPURandomGenerator::CPURandomGenerator() {
    std::random_device rd{};
    m_generator = std::mt19937(rd());
  }

  CPURandomGenerator::CPURandomGenerator(uint64_t seed) { m_generator = std::mt19937(seed); }

  int CPURandomGenerator::nrandi(int min, int max) {
    if (min > max)
      std::swap(min, max);
    std::uniform_int_distribution<int>::param_type dist(min, max);
    m_int_distrib.param(dist);
    return m_int_distrib(m_generator);
  }

  double CPURandomGenerator::nrandf(double min, double max) {
    if (min > max)
      std::swap(min, max);
    std::uniform_real_distribution<double>::param_type dist(min, max);
    m_float_distrib.param(dist);
    return m_float_distrib(m_generator);
  }

  bool CPURandomGenerator::randb() { return nrandi(0, 1); }

  /*
#if defined(AXOMAE_USE_CUDA)
  namespace gpgpu = device::gpgpu;
  ax_device_callable curandState_t *device_curand_states = nullptr;

  ax_kernel static void kernel_rand_init(curandState_t *states, uint64_t seed) {
    int idx = ax_linearCM2D_idx;
    curand_init(seed, 0, 0, &states[idx]);
  }

  void init_rand_device(const kernel_argpack_t &argpack, uint64_t seed) {
    std::size_t thread_number = (argpack.block_size.x * argpack.num_blocks.x) * (argpack.block_size.y * argpack.num_blocks.y) *
                                (argpack.block_size.z * argpack.num_blocks.z);
    auto curands_states = gpgpu::allocate_buffer(thread_number * sizeof(curandState_t));
    DEVICE_ERROR_CHECK(curands_states.error_status);
    exec_kernel(argpack, kernel_rand_init, static_cast<curandState_t *>(curands_states.device_ptr), seed);
    gpgpu::synchronize_device();
  }

  void clean_rand_device() { DEVICE_ERROR_CHECK(gpgpu::deallocate_buffer(device_curand_states).error_status); }

#endif

  int nrandi(int n1, int n2) {
#ifndef __CUDA_ARCH__
    auto &gen = get_generator();
    auto distrib = getUniformIntDistrib(n1, n2);
    return distrib(gen);
#else
    int idx = ax_linearCM2D_idx;
    curandState state = device_curand_states[idx];
    return n1 + (int)(curand_uniform(&state) * (n2 - n1 + 1));
#endif
  }

  double nrandf(double n1, double n2) {
#ifndef __CUDA_ARCH__
    auto &gen = get_generator();
    auto distrib = getUniformDoubleDistrib(n1, n2);
    return distrib(gen);
#else
    int idx = ax_linearCM2D_idx;
    curandState state = device_curand_states[idx];
    return n1 + (n2 - n1) * curand_uniform(&state);
#endif
  }*/

  void init_rand() {}
  void clean_rand() {}
}  // namespace math::random