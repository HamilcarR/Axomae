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
#endif
