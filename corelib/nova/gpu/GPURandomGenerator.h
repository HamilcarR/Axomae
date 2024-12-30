#ifndef GPURANDOMGENERATOR_H
#define GPURANDOMGENERATOR_H
#include "internal/common/math/math_random_interface.h"
#include <internal/device/gpgpu/device_utils.h>
#include <internal/macro/project_macros.h>

/* Generates random numbers specifically for the GPU .
 * There's two generators :
 * 1) A xor shift algorithm for fast pseudo random generation
 * 2) A sobol sequence generation for low discrepancy sampling.
 * Hammersley later.
 */

struct curandStateXORWOW;
struct curandStateScrambledSobol32;
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

  class GPUQuasiRandomGenerator : public AbstractRandomGenerator<GPUQuasiRandomGenerator> {
    /* Pointer to an array of states */
    curandStateScrambledSobol32 *device_curand_states{};
    uint64_t dimension{};
    std::size_t state_buffer_size{0};

   public:
    CLASS_DCM(GPUQuasiRandomGenerator)

    ax_device_callable explicit GPUQuasiRandomGenerator(curandStateScrambledSobol32 *curand_states_buffer, unsigned dimension = 3);
    ax_device_only int nrandi(int min, int max);
    ax_device_only float nrandf(float min, float max);
    ax_device_only glm::vec3 nrand3f(float min, float max);
    ax_device_only bool randb();

    ax_host_only void init(const kernel_argpack_t &kernel_config, unsigned dimension = 3);
    ax_host_only void cleanStates();
  };

}  // namespace math::random
#endif  // GPURANDOMGENERATOR_H
