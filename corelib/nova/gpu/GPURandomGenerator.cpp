#include "GPURandomGenerator.h"
#include "internal/device/gpgpu/device_utils.h"

#include <curand.h>
#include <curand_kernel.h>
#include <device_launch_parameters.h>
#include <internal/debug/debug_utils.h>
#include <internal/device/gpgpu/DeviceError.h>
#include <internal/device/gpgpu/device_transfer_interface.h>
#include <internal/device/gpgpu/kernel_launch_interface.h>

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

  ax_device_callable GPUQuasiRandomGenerator::GPUQuasiRandomGenerator(curandStateScrambledSobol32 *curand_states_buffer_, unsigned dimension_)
      : device_curand_states(curand_states_buffer_), dimension(dimension_) {
    if (dimension_ < 3)
      dimension = 3;
  }

  ax_device_only int GPUQuasiRandomGenerator::nrandi(int min, int max) {
    uint64_t idx = ax_device_linearCM3D_idx * dimension;
    if (idx >= state_buffer_size)
      idx = 0;
    curandStateScrambledSobol32_t &state = device_curand_states[idx];
    int rand = (int)curand_uniform(&state);
    int eval = to_interval(min, max, rand);
    return eval;
  }

  ax_device_only float GPUQuasiRandomGenerator::nrandf(float min, float max) {
    uint64_t idx = ax_device_linearCM3D_idx * dimension;
    if (idx >= state_buffer_size)
      idx = 0;
    curandStateScrambledSobol32_t &state = device_curand_states[idx];
    float rand = curand_uniform(&state);
    float eval = to_interval(min, max, rand);
    return eval;
  }

  ax_device_only glm::vec3 GPUQuasiRandomGenerator::nrand3f(float min, float max) {
    uint64_t idx = ax_device_linearCM3D_idx * dimension;
    glm::vec3 eval{};
    eval.x = to_interval(min, max, curand_uniform(&device_curand_states[idx]));
    eval.y = to_interval(min, max, curand_uniform(&device_curand_states[idx + 1]));
    eval.z = to_interval(min, max, curand_uniform(&device_curand_states[idx + 2]));
    return eval;
  }

  ax_device_only bool GPUQuasiRandomGenerator::randb() { return nrandi(0, 1) == 1; }

  ax_kernel void kernel_init_rand(curandStateScrambledSobol32_t *states,
                                  curandDirectionVectors32_t *vectors32,
                                  uint32_t *scramble,
                                  unsigned dimension) {
    uint64_t idx = ax_device_linearCM3D_idx * dimension;
    for (int i = 0; i < dimension; i++) {
      curand_init(vectors32[i], scramble[i], ax_device_linearCM2D_idx, &states[idx + i]);
    }
  }

  ax_host_only ax_dbg_optimize0 void GPUQuasiRandomGenerator::init(const kernel_argpack_t &argpack, unsigned dim) {
    std::size_t thread_number = argpack.computeThreadsNumber();
    if (dim < 3)
      dim = 3;
    dimension = dim;
    state_buffer_size = thread_number * dimension;
    auto curands_states = gpgpu::allocate_buffer(state_buffer_size * sizeof(curandStateScrambledSobol32_t));
    DEVICE_ERROR_CHECK(curands_states.error_status);
    device_curand_states = static_cast<curandStateScrambledSobol32_t *>(curands_states.device_ptr);

    auto device_direction_vectors = gpgpu::allocate_buffer(dimension * sizeof(curandDirectionVectors32_t));
    DEVICE_ERROR_CHECK(device_direction_vectors.error_status);
    curandDirectionVectors32_t *host_direction_vectors = nullptr;
    CUDA_ERROR_CHECK(curandGetDirectionVectors32(&host_direction_vectors, CURAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6));
    DEVICE_ERROR_CHECK(
        gpgpu::copy_buffer(
            host_direction_vectors, device_direction_vectors.device_ptr, dimension * sizeof(curandDirectionVectors32_t), gpgpu::HOST_DEVICE)
            .error_status);

    auto device_scramble_cste = gpgpu::allocate_buffer(dimension * sizeof(uint32_t));
    DEVICE_ERROR_CHECK(device_scramble_cste.error_status);
    uint32_t *host_scramble_cste = nullptr;
    CUDA_ERROR_CHECK(curandGetScrambleConstants32(&host_scramble_cste));
    DEVICE_ERROR_CHECK(
        gpgpu::copy_buffer(host_scramble_cste, device_scramble_cste.device_ptr, dimension * sizeof(uint32_t), gpgpu::HOST_DEVICE).error_status);
    exec_kernel(argpack,
                kernel_init_rand,
                device_curand_states,
                static_cast<curandDirectionVectors32_t *>(device_direction_vectors.device_ptr),
                static_cast<uint32_t *>(device_scramble_cste.device_ptr),
                dimension);
    gpgpu::synchronize_device();
    DEVICE_ERROR_CHECK(gpgpu::deallocate_buffer(device_direction_vectors.device_ptr).error_status);
    DEVICE_ERROR_CHECK(gpgpu::deallocate_buffer(device_scramble_cste.device_ptr).error_status);
  }

  ax_host_only void GPUQuasiRandomGenerator::cleanStates() {
    if (!device_curand_states)
      return;
    DEVICE_ERROR_CHECK(gpgpu::deallocate_buffer(device_curand_states).error_status);
    device_curand_states = nullptr;
  }

}  // namespace math::random