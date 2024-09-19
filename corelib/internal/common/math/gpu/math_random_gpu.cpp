
// clang-format off
#include "math_random_gpu.h"
#include <internal/device/gpgpu/device_utils.h>
#include <curand.h>
#include <curand_kernel.h>
#include <internal/device/gpgpu/DeviceError.h>
#include <internal/device/gpgpu/device_transfer_interface.h>
#include <internal/device/gpgpu/kernel_launch_interface.h>
// clang-format on

namespace gpgpu = device::gpgpu;
namespace math::random {

  ax_device_callable GPUPseudoRandomGenerator::GPUPseudoRandomGenerator(curandState_t *curand_states_buffer, std::size_t size, uint64_t seed_)
      : seed(seed_), state_buffer_size(size), device_curand_states(curand_states_buffer) {}

  ax_device_only int GPUPseudoRandomGenerator::nrandi(int min, int max) {
    uint64_t idx = ax_device_linearCM3D_idx;
    if (idx >= state_buffer_size)
      idx = 0;
    curandState_t &state = device_curand_states[idx];
    int eval = to_interval(min, max, curand_uniform(&state));
    return eval;
  }

  ax_device_only float GPUPseudoRandomGenerator::nrandf(float min, float max) {
    uint64_t idx = ax_device_linearCM3D_idx;
    if (idx >= state_buffer_size)
      idx = 0;
    curandState &state = device_curand_states[idx];
    float eval = to_interval(min, max, curand_uniform(&state));
    return eval;
  }

  ax_device_only glm::vec3 GPUPseudoRandomGenerator::nrand3f(float min, float max) { return {nrandf(min, max), nrandf(min, max), nrandf(min, max)}; }

  ax_device_only bool GPUPseudoRandomGenerator::randb() { return nrandi(0, 1) == 1; }

  ax_kernel void kernel_init_pseudo(curandState_t *states, uint64_t seed) {
    uint64_t idx = ax_device_linearCM3D_idx;
    curand_init(seed % idx, 0, idx, &states[idx]);
  }

  ax_host_only void GPUPseudoRandomGenerator::init(const kernel_argpack_t &argpack, uint64_t seed) {
    state_buffer_size = argpack.computeThreadsNumber();
    auto curands_states = gpgpu::allocate_buffer(state_buffer_size * sizeof(curandState_t));
    DEVICE_ERROR_CHECK(curands_states.error_status);
    device_curand_states = static_cast<curandState_t *>(curands_states.device_ptr);
    exec_kernel(argpack, kernel_init_pseudo, device_curand_states, seed);
    gpgpu::synchronize_device();
  }

  ax_host_only void GPUPseudoRandomGenerator::cleanStates() {
    if (!device_curand_states)
      return;
    DEVICE_ERROR_CHECK(gpgpu::deallocate_buffer(device_curand_states).error_status);
    device_curand_states = nullptr;
  }
  /*******************************************************************************************************************************************************************************/

}  // namespace math::random
