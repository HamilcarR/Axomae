#ifndef OPTIX_INTERNAL_H
#define OPTIX_INTERNAL_H
#include "aggregate/device_acceleration_interface.h"
#include "aggregate_datastructures.h"
#include "optix_types.h"
#include <optix_host.h>

//
// Internal header file. Don't include from outside the api/ folder.
//

#define OPTIX_ERR_CHECK(arg) \
  do { \
    OptixResult result = arg; \
    if (result != OPTIX_SUCCESS) \
      LOG("Optix error from call: " + std::string(#arg) + " Returned: " + std::to_string(result), LogLevel::ERROR); \
  } while (0)
;

namespace nova::aggregate {

  struct device_geometry_s {
    void *d_vertices{};
    void *d_indices{};
    void *d_transform{};
    /* In elements */
    std::size_t indices_size{};
    std::size_t vertices_size{};
    std::size_t transform_size{};
  };

  struct device_program_s {
    void *program_raygen;
    void *program_closest;
    void *program_miss;
  };

  struct device_pointers_s {
    device_geometry_s geometry;
    device_program_s program{};
  };

  // Use the make() method of DeviceAcceleratorInterface to generate an instance of BackendOptix.
  class BackendOptix : public DeviceAcceleratorInterface {
    OptixDeviceContext context;

   public:
    BackendOptix();
    AcceleratorHandle build(primitive_aggregate_data_s primitive_data_list) override;
    void cleanup() override;
    std::vector<OptixBuildInput> generateTrimeshBInputs(const primitive_aggregate_data_s &primitive_data_list);
    OptixProgramGroup createProgramGroup();
    OptixShaderBindingTable generateSbt(device_program_s &pointers);
  };

}  // namespace nova::aggregate

#endif  // OPTIX_INTERNAL_H
