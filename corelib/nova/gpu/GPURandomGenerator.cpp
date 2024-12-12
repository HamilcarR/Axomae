#include "GPURandomGenerator.h"
#include "internal/device/gpgpu/device_utils.h"
#include <curand.h>
#include <curand_kernel.h>
#include <device_launch_parameters.h>
#include <internal/device/gpgpu/DeviceError.h>
#include <internal/device/gpgpu/device_transfer_interface.h>
#include <internal/device/gpgpu/kernel_launch_interface.h>

namespace gpgpu = device::gpgpu;
namespace math::random {

  ax_device_only GPUPseudoRandomGenerator::GPUPseudoRandomGenerator(curandState_t *curand_states_buffer, uint64_t seed_)
      : seed(seed_), device_curand_states(curand_states_buffer) {}

  ax_device_only int GPUPseudoRandomGenerator::nrandi(int min, int max) {
    int idx = ax_linearCM3D_idx;
    curandState_t &state = device_curand_states[idx];
    int eval = to_interval(min, max, curand_uniform(&state));
    return eval;
  }

  ax_device_only double GPUPseudoRandomGenerator::nrandf(double min, double max) {
    int idx = ax_linearCM3D_idx;
    curandState &state = device_curand_states[idx];
    double eval = to_interval(min, max, curand_uniform(&state));
    return eval;
  }

  ax_device_only bool GPUPseudoRandomGenerator::randb() { return nrandi(0, 1) == 1; }

  ax_kernel void kernel_init_pseudo(curandState_t *states, uint64_t seed) {
    int idx = ax_linearCM3D_idx;
    curand_init(seed % idx, 0, 0, &states[idx]);
  }

  ax_host_only prand_alloc_result_t GPUPseudoRandomGenerator::init(const kernel_argpack_t &argpack, uint64_t seed) {
    std::size_t thread_number = (argpack.block_size.x * argpack.num_blocks.x) * (argpack.block_size.y * argpack.num_blocks.y) *
                                (argpack.block_size.z * argpack.num_blocks.z);
    auto curands_states = gpgpu::allocate_buffer(thread_number * sizeof(curandState_t));
    DEVICE_ERROR_CHECK(curands_states.error_status);
    exec_kernel(argpack, kernel_init_pseudo, static_cast<curandState_t *>(curands_states.device_ptr), seed);
    gpgpu::synchronize_device();
    return {static_cast<curandState_t *>(curands_states.device_ptr), seed, thread_number};
  }

  ax_host_only void GPUPseudoRandomGenerator::cleanStates(curandState_t *device_curand_states) {
    DEVICE_ERROR_CHECK(gpgpu::deallocate_buffer(device_curand_states).error_status);
  }
  /*******************************************************************************************************************************************************************************/

  ax_device_only GPUQuasiRandomGenerator::GPUQuasiRandomGenerator(curandStateScrambledSobol32 *curand_states_buffer_, unsigned dimension_)
      : device_curand_states(curand_states_buffer_), dimension(dimension_) {
    AX_ASSERT_LT(dimension_, 32);
  }

  ax_device_only int GPUQuasiRandomGenerator::nrandi(int min, int max) {
    int idx = ax_linearCM3D_idx * dimension;
    curandStateScrambledSobol32_t &state = device_curand_states[idx];
    int rand = (int)curand_uniform(&state);
    int eval = to_interval(min, max, rand);
    return eval;
  }

  ax_device_only double GPUQuasiRandomGenerator::nrandf(double min, double max) {
    uint64_t idx = ax_linearCM3D_idx;
    curandStateScrambledSobol32_t &state = device_curand_states[idx];
    double rand = curand_uniform(&state);
    double eval = to_interval(min, max, rand);
    return eval;
  }

  ax_device_only glm::vec3 GPUQuasiRandomGenerator::nrand3f(double min, double max) {
    uint64_t idx = ax_linearCM3D_idx;
    glm::vec3 eval{};
    eval.x = to_interval(min, max, curand_uniform(&device_curand_states[idx]));
    eval.y = to_interval(min, max, curand_uniform(&device_curand_states[idx + 1]));
    eval.z = to_interval(min, max, curand_uniform(&device_curand_states[idx + 2]));
    return eval;
  }

  ax_device_only bool GPUQuasiRandomGenerator::randb() { return nrandi(0, 1) == 1; }

  ax_kernel void kernel_init_rand(curandStateScrambledSobol32_t *states, curandDirectionVectors32_t *vectors32, uint32_t *scramble, int dimension) {
    uint32_t idx = ax_linearCM3D_idx * dimension;
    for (int i = 0; i < dimension; i++) {
      curand_init(vectors32[i], scramble[i], idx + i, &states[idx + i]);
    }
  }

  ax_host_only qrand_alloc_result_t GPUQuasiRandomGenerator::init(const kernel_argpack_t &argpack, unsigned dimension) {
    std::size_t thread_number = (argpack.block_size.x * argpack.num_blocks.x) * (argpack.block_size.y * argpack.num_blocks.y) *
                                (argpack.block_size.z * argpack.num_blocks.z);
    AX_ASSERT_LT(dimension, 32);
    auto curands_states = gpgpu::allocate_buffer(thread_number * dimension * sizeof(curandStateScrambledSobol32_t));
    DEVICE_ERROR_CHECK(curands_states.error_status);

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
                static_cast<curandStateScrambledSobol32_t *>(curands_states.device_ptr),
                static_cast<curandDirectionVectors32_t *>(device_direction_vectors.device_ptr),
                static_cast<uint32_t *>(device_scramble_cste.device_ptr),
                dimension);
    gpgpu::synchronize_device();

    DEVICE_ERROR_CHECK(gpgpu::deallocate_buffer(device_direction_vectors.device_ptr).error_status);
    DEVICE_ERROR_CHECK(gpgpu::deallocate_buffer(device_scramble_cste.device_ptr).error_status);
    return {static_cast<curandStateScrambledSobol32_t *>(curands_states.device_ptr), dimension, thread_number};
  }

  ax_host_only void GPUQuasiRandomGenerator::cleanStates(curandStateScrambledSobol32 *ptr_on_curands) {
    DEVICE_ERROR_CHECK(gpgpu::deallocate_buffer(ptr_on_curands).error_status);
  }

}  // namespace math::random