#include "aggregate_datastructures.h"
#include "glm/gtc/type_ptr.hpp"
#include "internal.h"
#include "optix_types.h"
#include "shape/MeshContext.h"
#include "shape/shape_datastructures.h"
#include <cstring>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <internal/debug/Logger.h>
#include <internal/device/gpgpu/DeviceError.h>
#include <internal/device/gpgpu/cuda/cuda_macros.h>
#include <internal/device/gpgpu/device_transfer_interface.h>
#include <internal/macro/project_macros.h>
#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_host.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>

constexpr std::size_t LOGSIZE = 4096;
constexpr std::size_t MAX_REC_DEPTH = 8;

static_assert(ISTYPE(OptixTraversableHandle, unsigned long long), "OptixTraversableHandle type implementation changed.");

void callback_error_log(unsigned level, const char *tag, const char *message, void *cbdata) {
  LOG("Optix log: TAG:" + std::string(tag) + ", MESSAGE:" + std::string(message) + " LEVEL:" + std::to_string(level), LogLevel::INFO);
}

static void device_dealloc(const nova::aggregate::allocations_tracker_s &allocations) {
  for (void *ptr : allocations.d_buffers) {
    DEVICE_ERROR_CHECK(cudaFree(ptr));
  }
}

struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) raygen_record_s {
  __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
};

struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) miss_record_s {
  __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
};

struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) randomhit_record_s {
  __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
};

namespace nova::aggregate {

  unsigned BackendOptix::getMaxRecursiveDepth() const { return MAX_REC_DEPTH; }

  BackendOptix::~BackendOptix() {
    BackendOptix::cleanup();
    optixDeviceContextDestroy(context);
  }

  BackendOptix::BackendOptix() : context(nullptr) {
    device::gpgpu::init_driver_API();
    OPTIX_ERR_CHECK(optixInit());
    OptixDeviceContextOptions options = {};
#ifndef NDEBUG
    options.logCallbackLevel = 4;
    options.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL;
#endif
    options.logCallbackFunction = callback_error_log;
    OPTIX_ERR_CHECK(optixDeviceContextCreate(0, nullptr, &context));
  }

  static void copy2device(const axstd::span<float> &vertices_array,
                          const axstd::span<unsigned> &indices_array,
                          const float transform_rm[12],  // Reads only 12 elements, no need for the perspective vector here.
                          device_buffers_s &geometry,
                          allocations_tracker_s &device_allocations) {

    auto err = device::gpgpu::allocate_buffer(vertices_array.size() * sizeof(float));
    DEVICE_ERROR_CHECK(err.error_status);
    AX_ASSERT_NOTNULL(err.device_ptr);
    geometry.d_vertices = err.device_ptr;
    geometry.vertices_size = vertices_array.size();
    device_allocations.d_buffers.push_back(geometry.d_vertices);

    err = device::gpgpu::allocate_buffer(indices_array.size() * sizeof(unsigned));
    DEVICE_ERROR_CHECK(err.error_status);
    AX_ASSERT_NOTNULL(err.device_ptr);
    geometry.d_indices = err.device_ptr;
    geometry.indices_size = indices_array.size();
    device_allocations.d_buffers.push_back(geometry.d_indices);

    err = device::gpgpu::allocate_buffer(12 * sizeof(float));
    DEVICE_ERROR_CHECK(err.error_status);
    AX_ASSERT_NOTNULL(err.device_ptr);
    geometry.d_transform = err.device_ptr;
    geometry.transform_size = 12;
    device_allocations.d_buffers.push_back(geometry.d_transform);

    err = device::gpgpu::copy_buffer(vertices_array.data(), geometry.d_vertices, vertices_array.size() * sizeof(float), device::gpgpu::HOST_DEVICE);
    DEVICE_ERROR_CHECK(err.error_status);
    AX_ASSERT_NOTNULL(err.device_ptr);

    err = device::gpgpu::copy_buffer(indices_array.data(), geometry.d_indices, indices_array.size() * sizeof(unsigned), device::gpgpu::HOST_DEVICE);
    DEVICE_ERROR_CHECK(err.error_status);
    AX_ASSERT_NOTNULL(err.device_ptr);

    err = device::gpgpu::copy_buffer(transform_rm, geometry.d_transform, 12 * sizeof(float), device::gpgpu::HOST_DEVICE);
    DEVICE_ERROR_CHECK(err.error_status);
    AX_ASSERT_NOTNULL(err.device_ptr);
  }

  /*********************************************************************************************************************************************************/
  /* Programs */

  static OptixProgramGroup create_raygen_pg(OptixDeviceContext context,
                                            const char *entry_name,
                                            OptixModule module,
                                            allocations_tracker_s & /*device_allocations*/) {
    LOGS("Creating Optix raygen program...");
    OptixProgramGroupDesc programDescriptions = {};
    programDescriptions.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    programDescriptions.raygen.module = module;
    programDescriptions.raygen.entryFunctionName = entry_name;
    OptixProgramGroupOptions options = {};
    OptixProgramGroup programGroups = nullptr;
    char logString[LOGSIZE];
    std::size_t logStringSize = 0;
    OPTIX_ERR_CHECK(optixProgramGroupCreate(context, &programDescriptions, 1, &options, logString, &logStringSize, &programGroups));
    LOGS(logString);
    return programGroups;
  }

  static OptixProgramGroup create_miss_pg(OptixDeviceContext context,
                                          const char *entry_name,
                                          OptixModule module,
                                          allocations_tracker_s & /*device_allocations*/) {

    LOGS("Creating Optix miss program...");
    OptixProgramGroupDesc programDescriptions = {};
    programDescriptions.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    programDescriptions.miss.module = module;
    programDescriptions.miss.entryFunctionName = entry_name;
    OptixProgramGroupOptions options = {};
    OptixProgramGroup programGroups = nullptr;
    char logString[LOGSIZE];
    std::size_t logStringSize = 0;
    OPTIX_ERR_CHECK(optixProgramGroupCreate(context, &programDescriptions, 1, &options, logString, &logStringSize, &programGroups));
    LOGS(logString);
    return programGroups;
  }

  static OptixProgramGroup create_closest_pg(OptixDeviceContext context,
                                             const char *entry_name,
                                             OptixModule module,
                                             allocations_tracker_s & /*device_allocations*/) {
    LOGS("Creating Optix closesthit program...");
    OptixProgramGroupDesc programDescriptions = {};
    programDescriptions.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    programDescriptions.hitgroup.moduleCH = module;
    programDescriptions.hitgroup.entryFunctionNameCH = entry_name;
    OptixProgramGroupOptions options = {};
    OptixProgramGroup programGroups = nullptr;
    char logString[LOGSIZE];
    std::size_t logStringSize = 0;
    OPTIX_ERR_CHECK(optixProgramGroupCreate(context, &programDescriptions, 1, &options, logString, &logStringSize, &programGroups));
    LOGS(logString);
    return programGroups;
  }

  static OptixProgramGroup create_any_pg(OptixDeviceContext context,
                                         const char *entry_name,
                                         OptixModule module,
                                         allocations_tracker_s & /*device_allocations*/) {
    LOGS("Creating Optix anyhit program...");
    OptixProgramGroupDesc programDescriptions = {};
    programDescriptions.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    programDescriptions.hitgroup.moduleAH = module;
    programDescriptions.hitgroup.entryFunctionNameAH = entry_name;
    OptixProgramGroupOptions options = {};
    OptixProgramGroup programGroups = nullptr;
    char logString[LOGSIZE];
    std::size_t logStringSize = 0;
    OPTIX_ERR_CHECK(optixProgramGroupCreate(context, &programDescriptions, 1, &options, logString, &logStringSize, &programGroups));
    LOGS(logString);
    return programGroups;
  }

  /*********************************************************************************************************************************************************/
  /* Pipeline */

  static OptixPipelineCompileOptions create_pipeline_compile_options() {
    OptixPipelineCompileOptions opts = {};
#ifndef NDEBUG
    opts.exceptionFlags = OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH;
#endif
    opts.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    opts.numPayloadValues = 3;
    opts.numAttributeValues = 0;
    opts.usesMotionBlur = false;
    opts.usesPrimitiveTypeFlags = 0;
    opts.pipelineLaunchParamsVariableName = "params";
    return opts;
  }

  static OptixPipelineLinkOptions create_pipeline_link_options() {
    OptixPipelineLinkOptions opts = {};
    opts.maxTraceDepth = MAX_REC_DEPTH;
    return opts;
  }

  static void setup_stacks(OptixPipeline pipeline,
                           OptixDeviceContext context,
                           const OptixProgramGroup *programGroups,
                           unsigned numProgramGroups,
                           allocations_tracker_s &device_allocations) {

    OptixStackSizes stackSizes = {};
    for (unsigned i = 0; i < numProgramGroups; i++)
      OPTIX_ERR_CHECK(optixUtilAccumulateStackSizes(programGroups[i], &stackSizes, pipeline));

    unsigned directCallableStackSizeFromState = 0, directCallableStackSizeFromTraversal = 0, continuationStackSize = 0;
    optixUtilComputeStackSizes(
        &stackSizes, MAX_REC_DEPTH, 0, 0, &directCallableStackSizeFromTraversal, &directCallableStackSizeFromState, &continuationStackSize);
    optixPipelineSetStackSize(pipeline, directCallableStackSizeFromTraversal, directCallableStackSizeFromState, continuationStackSize, 1);
  }

  static OptixPipeline create_pipeline(OptixDeviceContext context,
                                       const OptixProgramGroup *programGroups,
                                       const OptixPipelineCompileOptions *pipelineCompileOptions,
                                       const OptixPipelineLinkOptions *pipelineLinkOptions,
                                       unsigned numProgramGroups,
                                       allocations_tracker_s &device_allocations) {
    LOGS("Creating Optix pipeline...");
    OptixPipeline pipeline = nullptr;
    char logString[LOGSIZE] = {'\0'};
    std::size_t logStringSize = 0;
    OPTIX_ERR_CHECK(optixPipelineCreate(
        context, pipelineCompileOptions, pipelineLinkOptions, programGroups, numProgramGroups, logString, &logStringSize, &pipeline));
    LOGS(logString);
    setup_stacks(pipeline, context, programGroups, numProgramGroups, device_allocations);
    return pipeline;
  }

  /*********************************************************************************************************************************************************/
  /* Module */
  static OptixModule create_optix_module(OptixDeviceContext context,
                                         const OptixPipelineCompileOptions *pipelineCompileOptions,
                                         const char *ptx,  // Pointer to the input code
                                         std::size_t ptx_size,
                                         allocations_tracker_s &device_allocations) {
    LOGS("Creating Optix module...");
    OptixModule module = {};
    char log_buffer[LOGSIZE] = {'\0'};
    std::size_t log_buffer_size = LOGSIZE;
    OptixModuleCompileOptions module_compile_options = {};
#ifndef NDEBUG
    module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
    module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
#else
    module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_3;
    module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;
#endif
    module_compile_options.maxRegisterCount = 0;
    module_compile_options.payloadTypes = 0;
    module_compile_options.numPayloadTypes = 0;

    OPTIX_ERR_CHECK(
        optixModuleCreate(context, &module_compile_options, pipelineCompileOptions, ptx, ptx_size, log_buffer, &log_buffer_size, &module));
    LOGS(log_buffer);

    return module;
  }

  /*********************************************************************************************************************************************************/
  /* Shader Binding table */
  static OptixShaderBindingTable create_sbt(OptixProgramGroup program,
                                            const raygen_record_s &raygen_record,
                                            const miss_record_s &miss_record,
                                            const randomhit_record_s &randomhit_record,
                                            allocations_tracker_s &sbt_device_allocs) {
    LOGS("Creating Optix shader binding table...");
    OptixShaderBindingTable sbt = {};
    void *d_raygen = {};
    DEVICE_ERROR_CHECK(cudaMalloc(&d_raygen, sizeof(raygen_record_s)));
    DEVICE_ERROR_CHECK(cudaMemcpy(d_raygen, &raygen_record, sizeof(raygen_record_s), cudaMemcpyHostToDevice));
    sbt_device_allocs.d_buffers.push_back(d_raygen);
    sbt.raygenRecord = reinterpret_cast<CUdeviceptr>(d_raygen);

    void *d_miss = {};
    DEVICE_ERROR_CHECK(cudaMalloc(&d_miss, sizeof(miss_record_s)));
    DEVICE_ERROR_CHECK(cudaMemcpy(d_miss, &miss_record, sizeof(miss_record_s), cudaMemcpyHostToDevice));
    sbt_device_allocs.d_buffers.push_back(d_miss);
    sbt.missRecordBase = reinterpret_cast<CUdeviceptr>(d_miss);
    sbt.missRecordCount = 1;
    sbt.missRecordStrideInBytes = sizeof(miss_record_s);

    void *d_random = {};
    DEVICE_ERROR_CHECK(cudaMalloc(&d_random, sizeof(randomhit_record_s)));
    DEVICE_ERROR_CHECK(cudaMemcpy(d_random, &randomhit_record, sizeof(randomhit_record_s), cudaMemcpyHostToDevice));
    sbt_device_allocs.d_buffers.push_back(d_random);
    sbt.hitgroupRecordBase = reinterpret_cast<CUdeviceptr>(d_random);
    sbt.hitgroupRecordCount = 1;
    sbt.hitgroupRecordStrideInBytes = sizeof(randomhit_record_s);

    return sbt;
  }
  /*********************************************************************************************************************************************************/

  static const unsigned flags[] = {OPTIX_GEOMETRY_FLAG_DISABLE_TRIANGLE_FACE_CULLING};  // Needed for dielectrics materials , we don't want any
                                                                                        // triangle culled because of it's orientation .

  static OptixBuildInput create_trimesh_build_input(device_buffers_s &geometry) {
    OptixBuildInput input = {};
    input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    OptixBuildInputTriangleArray &build_input_tri = input.triangleArray;

    // optix shenanigans: Assume CUdeviceptr and void* are interchangeable.
    build_input_tri.vertexBuffers = reinterpret_cast<const CUdeviceptr *>(&geometry.d_vertices);
    build_input_tri.numVertices = geometry.vertices_size / 3;
    build_input_tri.vertexStrideInBytes = sizeof(float3);
    build_input_tri.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;

    build_input_tri.indexBuffer = reinterpret_cast<CUdeviceptr>(geometry.d_indices);
    build_input_tri.numIndexTriplets = geometry.indices_size / 3;
    build_input_tri.indexStrideInBytes = sizeof(uint3);
    build_input_tri.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;

    build_input_tri.preTransform = reinterpret_cast<CUdeviceptr>(geometry.d_transform);
    build_input_tri.transformFormat = OPTIX_TRANSFORM_FORMAT_MATRIX_FLOAT12;

    build_input_tri.numSbtRecords = 1;
    build_input_tri.flags = flags;
    return input;
  }

  static std::vector<OptixBuildInput> generate_trimesh_build_inputs(const primitive_aggregate_data_s &primitive_data_list,
                                                                    allocations_tracker_s &device_allocations) {

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
      device_allocs_s device_pointers;
      copy2device(vertices_array, indices_array, flat_transform, device_pointers.geometry, device_allocations);
      OptixBuildInput input = create_trimesh_build_input(device_pointers.geometry);
      inputs.push_back(input);
    }
    return inputs;
  }

  static OptixTraversableHandle build_triangle_gas(OptixDeviceContext context,
                                                   const primitive_aggregate_data_s &primitive_data_list,
                                                   void **d_outbuffer) {
    LOGS("Start building Optix GAS.");
    OptixAccelBuildOptions options = {};
    // TODO: BVH compression
    options.buildFlags = OPTIX_BUILD_FLAG_NONE;
    options.operation = OPTIX_BUILD_OPERATION_BUILD;
    options.motionOptions.numKeys = 0;
    allocations_tracker_s allocations;
    std::vector<OptixBuildInput> build_inputs = generate_trimesh_build_inputs(primitive_data_list, allocations);

    OptixAccelBufferSizes buffer_sizes = {};
    OPTIX_ERR_CHECK(optixAccelComputeMemoryUsage(context, &options, build_inputs.data(), build_inputs.size(), &buffer_sizes));

    void *d_tmp = nullptr;
    DEVICE_ERROR_CHECK(cudaMalloc(&d_tmp, buffer_sizes.tempSizeInBytes));
    DEVICE_ERROR_CHECK(cudaMalloc(&d_outbuffer, buffer_sizes.outputSizeInBytes));

    OptixTraversableHandle handle{};
    OPTIX_ERR_CHECK(optixAccelBuild(context,
                                    0,
                                    &options,
                                    build_inputs.data(),
                                    build_inputs.size(),
                                    reinterpret_cast<CUdeviceptr>(d_tmp),
                                    buffer_sizes.tempSizeInBytes,
                                    reinterpret_cast<CUdeviceptr>(d_outbuffer),
                                    buffer_sizes.outputSizeInBytes,
                                    &handle,
                                    nullptr,
                                    0));
    allocations.d_buffers.push_back(d_tmp);
    device_dealloc(allocations);
    LOGS("Optix GAS built.");
    return handle;
  }

  extern "C" const char PTX_EMBEDDED[];

  OptixTraversableHandle BackendOptix::build(primitive_aggregate_data_s primitive_data_list) {
    OptixTraversableHandle gas = build_triangle_gas(context, primitive_data_list, &d_outbuffer);

    OptixPipelineCompileOptions pipelineCompileOptions = create_pipeline_compile_options();
    OptixPipelineLinkOptions pipelineLinkOptions = create_pipeline_link_options();
    std::size_t ptx_size = 0;
    OptixModule module = create_optix_module(context, &pipelineCompileOptions, PTX_EMBEDDED, strlen(PTX_EMBEDDED), module_allocs);
    const char *raygen_ptx_entry = "__raygen__main";
    OptixProgramGroup raygen = create_raygen_pg(context, raygen_ptx_entry, module, pipeline_allocs);
    const char *miss_ptx_entry = "__miss__sample_envmap";
    OptixProgramGroup miss = create_miss_pg(context, miss_ptx_entry, module, pipeline_allocs);
    const char *any_ptx_entry = "__anyhit__random_intersect";
    OptixProgramGroup any = create_any_pg(context, any_ptx_entry, module, pipeline_allocs);
    const char *closest_ptx_entry = "__closesthit__minimum_intersect";
    OptixProgramGroup closest = create_closest_pg(context, closest_ptx_entry, module, pipeline_allocs);
    return gas;
  }

  void BackendOptix::cleanup() {
    if (!d_outbuffer)
      return;
    DEVICE_ERROR_CHECK(cudaFree(d_outbuffer));
    d_outbuffer = nullptr;
  }

}  // namespace nova::aggregate
