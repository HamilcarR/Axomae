#ifndef OPTIX_PARAMS_H
#define OPTIX_PARAMS_H
#include "nova_gpu.h"
#include <optix_types.h>

namespace nova {
  struct optix_traversal_param_s {
    OptixTraversableHandle handle{};
    nova::device_traversal_param_s d_params{};
  };
}  // namespace nova

struct path_payload_s {
  uint64_t prim_idx;
  float t, u, v;
  uint32_t traversal_stopped;  // This is a boolean, just need to explicitly pack it into a 32 bits register.
  float normal_matrix[9];
};

// Computes the number of required 32 bits registers for specific payload.
template<class PAYLOAD>
struct num_registers {
  static_assert((sizeof(PAYLOAD) / 4) < 32, "Provided payload is too big. ( > 128 bytes )");
  static constexpr size_t value = sizeof(PAYLOAD) / 4;
};

template<size_t NUM_REGISTERS = num_registers<path_payload_s>::value>
using register_stack_t = uint32_t[NUM_REGISTERS];

#endif
