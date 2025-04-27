#include "aggregate/device_acceleration_interface.h"
#include "gpu/nova_gpu.h"
#include "internal.h"
#include <cuda_runtime_api.h>
#include <internal/device/gpgpu/DeviceError.h>
#include <optix.h>
#include <optix_stubs.h>
namespace nova::aggregate {

  OptixIntersector::OptixIntersector(OptixTraversableHandle handle_id,
                                     OptixPipeline pipeline_,
                                     CUstream stream_,
                                     const OptixShaderBindingTable *sbt_,
                                     CUdeviceptr params_device_buffer)
      : accelerator(handle_id), pipeline(pipeline_), stream(stream_), sbt(sbt_), params_buffer(params_device_buffer) {}

  void OptixIntersector::traverse(const device_traversal_param_s &params) const {
    OPTIX_ERR_CHECK(optixLaunch(pipeline, stream, params_buffer, sizeof(device_traversal_param_s), sbt, params.width, params.height, params.depth));
    DEVICE_ERROR_CHECK(cudaDeviceSynchronize());
  }

}  // namespace nova::aggregate
