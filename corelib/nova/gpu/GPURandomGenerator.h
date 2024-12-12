#ifndef GPURANDOMGENERATOR_H
#define GPURANDOMGENERATOR_H
#include "internal/common/math/math_random.h"

#include <glm/vec3.hpp>

/* Generates random numbers specifically for the GPU .
 * There's two generators :
 * 1) A xor shift algorithm for fast pseudo random generation
 * 2) A sobol sequence generation for low discrepancy sampling.
 * Not gonna bother with Hammersley.
 */

struct curandStateXORWOW;
struct curandStateScrambledSobol32;
struct kernel_argpack_t;

namespace math::random {

  struct prand_alloc_result_t {
    curandStateXORWOW *states_array{};
    uint64_t seed{};
    std::size_t num_threads{};
  };

  class GPUPseudoRandomGenerator : public AbstractRandomGenerator<GPUPseudoRandomGenerator> {
    /* Pointer to an array of states */
    curandStateXORWOW *device_curand_states{};
    uint64_t seed{1};

   public:
    CLASS_DCM(GPUPseudoRandomGenerator)

    ax_device_only GPUPseudoRandomGenerator(curandStateXORWOW *curand_states_buffer, uint64_t seed);
    ax_device_only int nrandi(int min, int max);
    ax_device_only double nrandf(double min, double max);
    ax_device_only bool randb();

    ax_host_only static prand_alloc_result_t init(const kernel_argpack_t &kernel_config, uint64_t seed);
    ax_host_only static void cleanStates(curandStateXORWOW *device_ptr);
  };

  struct qrand_alloc_result_t {
    curandStateScrambledSobol32 *states_array{};
    unsigned dimension{};
    std::size_t num_threads{};
  };

  class GPUQuasiRandomGenerator : public AbstractRandomGenerator<GPUQuasiRandomGenerator> {
    /* Pointer to an array of states */
    curandStateScrambledSobol32 *device_curand_states{};
    unsigned dimension{1};

   public:
    CLASS_DCM(GPUQuasiRandomGenerator)

    ax_device_only GPUQuasiRandomGenerator(curandStateScrambledSobol32 *curand_states_buffer, unsigned dimension);
    ax_device_only int nrandi(int min, int max);
    ax_device_only double nrandf(double min, double max);
    ax_device_only glm::vec3 nrand3f(double min, double max);
    ax_device_only bool randb();

    ax_host_only static qrand_alloc_result_t init(const kernel_argpack_t &kernel_config, unsigned dimension);
    ax_host_only static void cleanStates(curandStateScrambledSobol32 *device_ptr);
  };

}  // namespace math::random
#endif  // GPURANDOMGENERATOR_H
