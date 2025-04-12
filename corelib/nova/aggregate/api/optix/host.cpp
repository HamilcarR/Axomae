#include "aggregate_datastructures.h"
#include "glm/gtc/type_ptr.hpp"
#include "internal.h"
#include "optix_types.h"
#include "shape/MeshContext.h"
#include "shape/shape_datastructures.h"
#include <cstring>
#include <internal/debug/Logger.h>
#include <internal/device/gpgpu/DeviceError.h>
#include <internal/device/gpgpu/device_transfer_interface.h>
#include <internal/macro/project_macros.h>
#include <optix.h>
#include <optix_function_table_definition.h>  //Needed to link optix functions to CUDA driver
#include <optix_host.h>
#include <optix_stubs.h>

static_assert(ISTYPE(OptixTraversableHandle, unsigned long long), "OptixTraversableHandle type implementation changed.");

void callback_error_log(unsigned level, const char *tag, const char *message, void *cbdata) {
  LOG("Optix log: TAG:" + std::string(tag) + ", MESSAGE:" + std::string(message) + " LEVEL:" + std::to_string(level), LogLevel::INFO);
}

struct empty_record_s {};

template<class T>
struct sbt_record_s {
  __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
  T data;
};

using SbtRecord = sbt_record_s<bvh_hit_data>;

namespace nova::aggregate {

  BackendOptix::BackendOptix() : context(nullptr) {
    device::gpgpu::init_driver_API();
    optixInit();
    OptixDeviceContextOptions options = {};
#ifndef NDEBUG
    options.logCallbackLevel = 4;
    options.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL;
#endif
    options.logCallbackFunction = callback_error_log;
    optixDeviceContextCreate(0, nullptr, &context);
  }

  static device_pointers_s copy2device(const axstd::span<float> &vertices_array,
                                       const axstd::span<unsigned> &indices_array,
                                       const float transform_rm[12])  // Reads only 12 elements, no need for the perspective vector here.
  {

    device_pointers_s return_dev_ptr = {};
    auto err = device::gpgpu::allocate_buffer(vertices_array.size() * sizeof(float));
    DEVICE_ERROR_CHECK(err.error_status);
    AX_ASSERT_NOTNULL(err.device_ptr);
    return_dev_ptr.geometry.d_vertices = err.device_ptr;
    return_dev_ptr.geometry.vertices_size = vertices_array.size();

    err = device::gpgpu::allocate_buffer(indices_array.size() * sizeof(unsigned));
    DEVICE_ERROR_CHECK(err.error_status);
    AX_ASSERT_NOTNULL(err.device_ptr);
    return_dev_ptr.geometry.d_indices = err.device_ptr;
    return_dev_ptr.geometry.indices_size = indices_array.size();

    err = device::gpgpu::allocate_buffer(12 * sizeof(float));
    DEVICE_ERROR_CHECK(err.error_status);
    AX_ASSERT_NOTNULL(err.device_ptr);
    return_dev_ptr.geometry.d_transform = err.device_ptr;
    return_dev_ptr.geometry.transform_size = 12;

    err = device::gpgpu::copy_buffer(
        vertices_array.data(), return_dev_ptr.geometry.d_vertices, vertices_array.size() * sizeof(float), device::gpgpu::HOST_DEVICE);
    DEVICE_ERROR_CHECK(err.error_status);
    AX_ASSERT_NOTNULL(err.device_ptr);

    err = device::gpgpu::copy_buffer(
        indices_array.data(), return_dev_ptr.geometry.d_indices, indices_array.size() * sizeof(unsigned), device::gpgpu::HOST_DEVICE);
    DEVICE_ERROR_CHECK(err.error_status);
    AX_ASSERT_NOTNULL(err.device_ptr);

    err = device::gpgpu::copy_buffer(transform_rm, return_dev_ptr.geometry.d_transform, 12 * sizeof(float), device::gpgpu::HOST_DEVICE);
    DEVICE_ERROR_CHECK(err.error_status);
    AX_ASSERT_NOTNULL(err.device_ptr);

    return return_dev_ptr;
  }

  OptixProgramGroup BackendOptix::createProgramGroup() { return {}; }

  OptixShaderBindingTable BackendOptix::generateSbt(device_program_s &pointers) {
    OptixShaderBindingTable sbt = {};
    sbt_record_s<empty_record_s> empty = {};
    OptixProgramGroup raygen_pg = {}, miss_pg = {}, closest_hit_pg = {};
    optixSbtRecordPackHeader(raygen_pg, &empty);
    optixSbtRecordPackHeader(miss_pg, &empty);
    optixSbtRecordPackHeader(closest_hit_pg, &empty);

    return sbt;
  }

  static OptixBuildInput create_trimesh_input(device_pointers_s &pointers) {
    OptixBuildInput input = {};
    input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    OptixBuildInputTriangleArray &build_input_tri = input.triangleArray;

    // optix shenanigans: Assume CUdeviceptr and void* are interchangeable.
    build_input_tri.vertexBuffers = reinterpret_cast<const CUdeviceptr *>(&pointers.geometry.d_vertices);
    build_input_tri.numVertices = pointers.geometry.vertices_size / 3;
    build_input_tri.vertexStrideInBytes = sizeof(float3);
    build_input_tri.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;

    build_input_tri.indexBuffer = reinterpret_cast<CUdeviceptr>(pointers.geometry.d_indices);
    build_input_tri.numIndexTriplets = pointers.geometry.indices_size / 3;
    build_input_tri.indexStrideInBytes = sizeof(uint3);
    build_input_tri.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;

    build_input_tri.preTransform = reinterpret_cast<CUdeviceptr>(pointers.geometry.d_transform);
    build_input_tri.transformFormat = OPTIX_TRANSFORM_FORMAT_MATRIX_FLOAT12;

    build_input_tri.numSbtRecords = 1;

    return input;
  }

  std::vector<OptixBuildInput> BackendOptix::generateTrimeshBInputs(const primitive_aggregate_data_s &primitive_data_list) {
    std::vector<OptixBuildInput> inputs;
    const shape::MeshBundleViews &mesh_bundle = primitive_data_list.mesh_geometry;
    std::size_t total_triangle_meshes = mesh_bundle.triMeshCount();
    for (std::size_t mesh_index = 0; mesh_index < total_triangle_meshes; mesh_index++) {
      const Object3D &current_triangle_mesh = mesh_bundle.getTriangleMesh(mesh_index);
      const axstd::span<float> &vertices_array = current_triangle_mesh.vertices;
      const axstd::span<unsigned> &indices_array = current_triangle_mesh.indices;

      // Optix expects matrices to be represented in row major, so we use the cached transpose transform
      const glm::mat4 &transform_rm = mesh_bundle.reconstructTransform4x4(mesh_index).t;
      const float *flat_transform = glm::value_ptr(transform_rm);
      device_pointers_s device_pointers = copy2device(vertices_array, indices_array, flat_transform);

      OptixBuildInput input = create_trimesh_input(device_pointers);
      inputs.push_back(input);
    }
    return inputs;
  }

  OptixTraversableHandle BackendOptix::build(primitive_aggregate_data_s primitive_data_list) {
    OptixAccelBuildOptions options = {};
    options.buildFlags = OPTIX_BUILD_FLAG_NONE;
    options.operation = OPTIX_BUILD_OPERATION_BUILD;
    options.motionOptions.numKeys = 0;
    std::vector<OptixBuildInput> build_inputs = generateTrimeshBInputs(primitive_data_list);

    OptixAccelBufferSizes buffer_sizes = {};
    OPTIX_ERR_CHECK(optixAccelComputeMemoryUsage(context, &options, build_inputs.data(), build_inputs.size(), &buffer_sizes));

    void *temporary_alloc = nullptr, *output_alloc = nullptr;
    DEVICE_ERROR_CHECK(device::gpgpu::allocate_symbol(&temporary_alloc, buffer_sizes.tempSizeInBytes).error_status);
    DEVICE_ERROR_CHECK(device::gpgpu::allocate_symbol(&output_alloc, buffer_sizes.outputSizeInBytes).error_status);
    OptixTraversableHandle handle{};
    OPTIX_ERR_CHECK(optixAccelBuild(context,
                                    nullptr,
                                    &options,
                                    build_inputs.data(),
                                    build_inputs.size(),
                                    reinterpret_cast<CUdeviceptr>(temporary_alloc),
                                    buffer_sizes.tempSizeInBytes,
                                    reinterpret_cast<CUdeviceptr>(output_alloc),
                                    buffer_sizes.outputSizeInBytes,
                                    &handle,
                                    nullptr,
                                    0));
    return handle;
  }

  void BackendOptix::cleanup() {}

}  // namespace nova::aggregate
