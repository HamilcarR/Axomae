#ifndef MATH_RANDOM_GPU_H
#define MATH_RANDOM_GPU_H
#include "../math_random_interface.h"
#include <internal/device/gpgpu/device_utils.h>
#include <internal/macro/project_macros.h>

#ifdef AXOMAE_USE_CUDA

/* Generates random numbers specifically for the GPU .
 * There's two generators :
 * 1) A xor shift algorithm for fast pseudo random generation
 * 2) A sobol sequence generation for low discrepancy sampling.
 */

struct curandStateXORWOW;
struct kernel_argpack_t;

namespace math::random {

  class GPUPseudoRandomGenerator : public AbstractRandomGenerator<GPUPseudoRandomGenerator> {
    /* Pointer to an array of states */
    curandStateXORWOW *device_curand_states{};
    uint64_t seed{};
    std::size_t state_buffer_size{};

   public:
    CLASS_DCM(GPUPseudoRandomGenerator)

    ax_device_callable explicit GPUPseudoRandomGenerator(curandStateXORWOW *curand_states_buffer,
                                                         std::size_t state_buffer_size,
                                                         uint64_t seed = 0xDEADBEEF);
    ax_device_only int nrandi(int min, int max);
    ax_device_only float nrandf(float min, float max);
    ax_device_only glm::vec3 nrand3f(float min, float max);
    ax_device_only bool randb();

    ax_host_only void init(const kernel_argpack_t &kernel_config, uint64_t seed = 0xDEADBEEF);
    ax_host_only void cleanStates();
  };

}  // namespace math::random
#endif

#endif
