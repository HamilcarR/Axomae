#ifndef OPTIX_INTERNAL_H
#define OPTIX_INTERNAL_H
#include "aggregate/device_acceleration_interface.h"
#include "aggregate_datastructures.h"
#include "gpu/nova_gpu.h"
#include <optix_host.h>
#include <optix_types.h>

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

  class OptixAccelerator : public DeviceAcceleratorInterface {
    OptixDeviceContext context{};
    CUcontext cuctx{};
    allocations_tracker_s sbt_allocs{}, pipeline_allocs{}, module_allocs{};
    OptixShaderBindingTable intersect_sbt{};
    OptixProgramGroup programs[16]{};
    static constexpr int NUM_PROGRAMS = 3;
    OptixPipeline pipeline{};
    OptixTraversableHandle handle{};
    OptixModule module{};
    void *d_outbuffer{}, *d_params_buffer{};

   public:
    OptixAccelerator();
    ~OptixAccelerator() override;
    OptixAccelerator(const OptixAccelerator &) = default;
    OptixAccelerator(OptixAccelerator &&) noexcept = default;
    OptixAccelerator &operator=(const OptixAccelerator &) = default;
    OptixAccelerator &operator=(OptixAccelerator &&) noexcept = default;
    void build(primitive_aggregate_data_s primitive_data_list) override;
    void copyParamsToDevice(const device_traversal_param_s &params) const override;
    void cleanup() override;
    unsigned getMaxRecursiveDepth() const override;
    std::unique_ptr<DeviceIntersectorInterface> getIntersectorObject() const override;
  };

  class OptixIntersector final : public DeviceIntersectorInterface {
    OptixTraversableHandle accelerator{};
    OptixPipeline pipeline{};
    CUstream stream{};
    const OptixShaderBindingTable *sbt{};
    CUdeviceptr params_buffer;

   public:
    OptixIntersector(
        OptixTraversableHandle handle_id, OptixPipeline pipeline, CUstream stream, const OptixShaderBindingTable *sbt, CUdeviceptr params_buffer);
    void traverse(const device_traversal_param_s &params) const override;
  };

}  // namespace nova::aggregate

#endif  // OPTIX_INTERNAL_H
