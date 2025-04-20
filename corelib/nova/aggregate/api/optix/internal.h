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

  struct device_buffers_s {
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

  struct device_allocs_s {
    device_buffers_s geometry;
    device_program_s program{};
  };

  struct allocations_tracker_s {
    std::vector<void *> d_buffers;
  };

  // Use the make() method of DeviceAcceleratorInterface to generate an instance of BackendOptix.
  class BackendOptix final : public DeviceAcceleratorInterface {
    OptixDeviceContext context;
    allocations_tracker_s sbt_allocs, pipeline_allocs, module_allocs;
    OptixShaderBindingTable intersect_sbt, randomhit_sbt, miss_sbt;
    OptixPipeline pipeline;
    void *d_outbuffer{};

   public:
    BackendOptix();
    ~BackendOptix() override;
    BackendOptix(const BackendOptix &) = default;
    BackendOptix(BackendOptix &&) noexcept = default;
    BackendOptix &operator=(const BackendOptix &) = default;
    BackendOptix &operator=(BackendOptix &&) noexcept = default;
    AcceleratorHandle build(primitive_aggregate_data_s primitive_data_list) override;
    void cleanup() override;
    unsigned getMaxRecursiveDepth() const override;
  };

}  // namespace nova::aggregate

#endif  // OPTIX_INTERNAL_H
