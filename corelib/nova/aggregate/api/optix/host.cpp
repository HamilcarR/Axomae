#include "../../aggregate_datastructures.h"
#include "glm/matrix.hpp"
#include "gpu/optix_params.h"
#include "internal.h"
#include "shape/MeshContext.h"
#include "shape/shape_datastructures.h"
#include <cstring>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <internal/common/math/math_includes.h>
#include <internal/debug/Logger.h>
#include <internal/device/gpgpu/DeviceError.h>
#include <internal/device/gpgpu/device_transfer_interface.h>
#include <internal/macro/project_macros.h>
#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_host.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>
#include <optix_types.h>
#include <string>

constexpr std::size_t LOGSIZE = 4096;
constexpr std::size_t MAX_REC_DEPTH = 8;

static_assert(ISTYPE(OptixTraversableHandle, unsigned long long), "OptixTraversableHandle type implementation changed.");

void optix_callback_log(unsigned level, const char *tag, const char *message, void *cbdata) {
  LOG("Optix log: TAG:" + std::string(tag) + ", MESSAGE:" + std::string(message) + " LEVEL:" + std::to_string(level), LogLevel::INFO);
}

/* Free the memory on device, and clear the underlying d_buffers collection. */
static void device_dealloc(nova::aggregate::allocations_tracker_s &allocations) {
  for (void *ptr : allocations.d_buffers) {
    DEVICE_ERROR_CHECK(cudaFree(ptr));
  }
  allocations.d_buffers.clear();
}

struct raygen_record_s {
  __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
};

struct misshit_record_s {
  __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
};

struct hit_record_s {
  __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
};

struct exception_record_s {
  __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
};

namespace nova::aggregate {

  unsigned OptixAccelerator::getMaxRecursiveDepth() const { return MAX_REC_DEPTH; }

  OptixAccelerator::OptixAccelerator() {
    AX_ASSERT_EQ(cuCtxGetCurrent(&cuctx), CUDA_SUCCESS);
    DEVICE_ERROR_CHECK(cudaFree(0));
    OPTIX_ERR_CHECK(optixInit());
    OptixDeviceContextOptions options = {};
#ifndef NDEBUG
    options.logCallbackLevel = 4;
    options.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL;
#endif
    options.logCallbackFunction = optix_callback_log;
    OPTIX_ERR_CHECK(optixDeviceContextCreate(cuctx, &options, &context));
  }

  OptixAccelerator::~OptixAccelerator() {
    OptixAccelerator::cleanup();
    OPTIX_ERR_CHECK(optixDeviceContextDestroy(context));
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
    std::size_t logStringSize = LOGSIZE;
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
    std::size_t logStringSize = LOGSIZE;
    OPTIX_ERR_CHECK(optixProgramGroupCreate(context, &programDescriptions, 1, &options, logString, &logStringSize, &programGroups));
    LOGS(logString);
    return programGroups;
  }

  static OptixProgramGroup create_hit_pg(OptixDeviceContext context,
                                         const char *anyhit_entry,
                                         const char *closesthit_entry,
                                         OptixModule anyhit_module,
                                         OptixModule closesthit_module,
                                         allocations_tracker_s & /*device_allocations*/) {
    LOGS("Creating Optix anyhit program...");
    OptixProgramGroupDesc programDescriptions = {};
    programDescriptions.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;

    programDescriptions.hitgroup.moduleAH = anyhit_module;
    programDescriptions.hitgroup.entryFunctionNameAH = anyhit_entry;

    programDescriptions.hitgroup.moduleCH = closesthit_module;
    programDescriptions.hitgroup.entryFunctionNameCH = closesthit_entry;

    OptixProgramGroupOptions options = {};
    OptixProgramGroup programGroups = nullptr;
    char logString[LOGSIZE];
    std::size_t logStringSize = LOGSIZE;
    OPTIX_ERR_CHECK(optixProgramGroupCreate(context, &programDescriptions, 1, &options, logString, &logStringSize, &programGroups));
    LOGS(logString);
    return programGroups;
  }

  static OptixProgramGroup create_exception_pg(OptixDeviceContext context,
                                               const char *entry_name,
                                               OptixModule module,
                                               allocations_tracker_s & /*device_allocations*/) {

    LOGS("Creating Optix exception program...");
    OptixProgramGroupDesc programDescriptions = {};
    programDescriptions.kind = OPTIX_PROGRAM_GROUP_KIND_EXCEPTION;
    programDescriptions.exception.module = module;
    programDescriptions.exception.entryFunctionName = entry_name;
    OptixProgramGroupOptions options = {};
    OptixProgramGroup programGroups = nullptr;
    char logString[LOGSIZE];
    std::size_t logStringSize = LOGSIZE;
    OPTIX_ERR_CHECK(optixProgramGroupCreate(context, &programDescriptions, 1, &options, logString, &logStringSize, &programGroups));
    LOGS(logString);
    return programGroups;
  }

  /*********************************************************************************************************************************************************/
  /* Pipeline */

  static OptixPipelineCompileOptions create_pipeline_compile_options() {
    OptixPipelineCompileOptions opts = {};
#ifndef NDEBUG
    opts.exceptionFlags = OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH | OPTIX_EXCEPTION_FLAG_USER;
#endif
    opts.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    opts.numPayloadValues = num_registers<path_payload_s>::value;
    opts.numAttributeValues = 0;
    opts.usesMotionBlur = false;
    opts.usesPrimitiveTypeFlags = 0;
    opts.pipelineLaunchParamsVariableName = "parameters";
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
    OPTIX_ERR_CHECK(optixUtilComputeStackSizes(
        &stackSizes, MAX_REC_DEPTH, 0, 0, &directCallableStackSizeFromTraversal, &directCallableStackSizeFromState, &continuationStackSize));
    OPTIX_ERR_CHECK(
        optixPipelineSetStackSize(pipeline, directCallableStackSizeFromTraversal, directCallableStackSizeFromState, continuationStackSize, 1));
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
    std::size_t logStringSize = LOGSIZE;
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

    OPTIX_ERR_CHECK(
        optixModuleCreate(context, &module_compile_options, pipelineCompileOptions, ptx, ptx_size, log_buffer, &log_buffer_size, &module));
    LOGS(log_buffer);

    return module;
  }

  /*********************************************************************************************************************************************************/
  /* Shader Binding table */
  static OptixShaderBindingTable create_sbt(OptixProgramGroup raygen,
                                            OptixProgramGroup miss,
                                            OptixProgramGroup hit,
                                            OptixProgramGroup exception,
                                            raygen_record_s &raygen_record,
                                            misshit_record_s &misshit_record,
                                            hit_record_s &hit_record,
                                            exception_record_s &exception_record,
                                            allocations_tracker_s &sbt_device_allocs) {
    LOGS("Creating Optix shader binding table...");
    OptixShaderBindingTable sbt = {};

    OPTIX_ERR_CHECK(optixSbtRecordPackHeader(raygen, &raygen_record));
    void *d_raygen = {};
    DEVICE_ERROR_CHECK(cudaMalloc(&d_raygen, sizeof(raygen_record_s)));
    DEVICE_ERROR_CHECK(cudaMemcpy(d_raygen, &raygen_record, sizeof(raygen_record_s), cudaMemcpyHostToDevice));
    sbt_device_allocs.d_buffers.push_back(d_raygen);
    sbt.raygenRecord = reinterpret_cast<CUdeviceptr>(d_raygen);

    OPTIX_ERR_CHECK(optixSbtRecordPackHeader(miss, &misshit_record));
    void *d_miss = {};
    DEVICE_ERROR_CHECK(cudaMalloc(&d_miss, sizeof(misshit_record_s)));
    DEVICE_ERROR_CHECK(cudaMemcpy(d_miss, &misshit_record, sizeof(misshit_record_s), cudaMemcpyHostToDevice));
    sbt_device_allocs.d_buffers.push_back(d_miss);
    sbt.missRecordBase = reinterpret_cast<CUdeviceptr>(d_miss);
    sbt.missRecordCount = 1;
    sbt.missRecordStrideInBytes = sizeof(misshit_record_s);

    OPTIX_ERR_CHECK(optixSbtRecordPackHeader(hit, &hit_record));
    void *d_hit = {};
    DEVICE_ERROR_CHECK(cudaMalloc(&d_hit, sizeof(hit_record_s)));
    DEVICE_ERROR_CHECK(cudaMemcpy(d_hit, &hit_record, sizeof(hit_record_s), cudaMemcpyHostToDevice));
    sbt_device_allocs.d_buffers.push_back(d_hit);
    sbt.hitgroupRecordBase = reinterpret_cast<CUdeviceptr>(d_hit);
    sbt.hitgroupRecordCount = 1;
    sbt.hitgroupRecordStrideInBytes = sizeof(hit_record_s);

    OPTIX_ERR_CHECK(optixSbtRecordPackHeader(exception, &exception_record));
    void *d_exception = {};
    DEVICE_ERROR_CHECK(cudaMalloc(&d_exception, sizeof(exception_record_s)));
    DEVICE_ERROR_CHECK(cudaMemcpy(d_exception, &exception_record, sizeof(exception_record_s), cudaMemcpyHostToDevice));
    sbt_device_allocs.d_buffers.push_back(d_exception);
    sbt.exceptionRecord = reinterpret_cast<CUdeviceptr>(d_exception);

    return sbt;
  }
  /*********************************************************************************************************************************************************/

  static const unsigned flags[] = {OPTIX_GEOMETRY_FLAG_DISABLE_TRIANGLE_FACE_CULLING,  // Needed for dielectrics materials , we don't want any
                                                                                       // triangle culled because of it's orientation..
                                   OPTIX_RAY_FLAG_DISABLE_ANYHIT};

  struct alignas(16) aligned_vertex {
    float x, y, z, w;
  };

  static OptixBuildInput create_trimesh_build_input(const device_buffers_s &geometry, std::size_t num_vertices, std::size_t prim_offset) {
    OptixBuildInput input = {};
    input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    OptixBuildInputTriangleArray &build_input_tri = input.triangleArray;

    build_input_tri.vertexBuffers = reinterpret_cast<const CUdeviceptr *>(&geometry.d_vertices);
    build_input_tri.numVertices = num_vertices;
    build_input_tri.vertexStrideInBytes = sizeof(aligned_vertex);
    build_input_tri.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;

    AX_ASSERT(geometry.indices_size % 3 == 0, "Total indices number is non divisible by 3. (Not a triangle ?)");
    build_input_tri.indexBuffer = reinterpret_cast<CUdeviceptr>(geometry.d_indices);
    build_input_tri.numIndexTriplets = geometry.indices_size / 3;
    build_input_tri.indexStrideInBytes = 3 * sizeof(unsigned int);
    build_input_tri.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;

    build_input_tri.preTransform = reinterpret_cast<CUdeviceptr>(geometry.d_transform);
    build_input_tri.transformFormat = OPTIX_TRANSFORM_FORMAT_MATRIX_FLOAT12;

    build_input_tri.primitiveIndexOffset = prim_offset;
    build_input_tri.numSbtRecords = 1;
    build_input_tri.flags = flags;

    return input;
  }

  /* Allocates memory on gpu and copy vertices, indices, and transform matrices to it. */
  template<class C1, class C2>
  static void copy2device(const C1 &vertices_array,
                          const C2 &indices_array,
                          const float transform_rm[12],  // Reads only 12 elements, no need for the perspective vector here.
                          device_buffers_s &geometry,
                          allocations_tracker_s &device_allocations) {

    auto err = device::gpgpu::allocate_buffer(vertices_array.size() * sizeof(aligned_vertex));
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

    err = device::gpgpu::copy_buffer(
        vertices_array.data(), geometry.d_vertices, vertices_array.size() * sizeof(aligned_vertex), device::gpgpu::HOST_DEVICE);
    DEVICE_ERROR_CHECK(err.error_status);
    AX_ASSERT_NOTNULL(err.device_ptr);

    err = device::gpgpu::copy_buffer(indices_array.data(), geometry.d_indices, indices_array.size() * sizeof(unsigned), device::gpgpu::HOST_DEVICE);
    DEVICE_ERROR_CHECK(err.error_status);
    AX_ASSERT_NOTNULL(err.device_ptr);

    err = device::gpgpu::copy_buffer(transform_rm, geometry.d_transform, 12 * sizeof(float), device::gpgpu::HOST_DEVICE);
    DEVICE_ERROR_CHECK(err.error_status);
    AX_ASSERT_NOTNULL(err.device_ptr);
  }

  /* Returns a vector of quadruplets for 16 bytes padding. Needed for optimal build.  */
  inline std::vector<aligned_vertex> align_vertices(const axstd::span<float> &vertices) {
    std::vector<aligned_vertex> ret;
    std::size_t i = 0;
    for (i = 0; i < vertices.size(); i += 3) {
      aligned_vertex vert = {vertices[i], vertices[i + 1], vertices[i + 2], 0};
      ret.push_back(vert);
    }
    return ret;
  }

  static std::vector<OptixBuildInput> generate_trimesh_build_inputs(const primitive_aggregate_data_s &primitive_data_list,
                                                                    std::vector<device_allocs_s> &device_pointers_collection,
                                                                    allocations_tracker_s &device_allocations) {

    std::vector<OptixBuildInput> inputs;
    const shape::MeshBundleViews &mesh_bundle = primitive_data_list.mesh_geometry;
    std::size_t total_triangle_meshes = mesh_bundle.triMeshCount();
    std::size_t primitive_offset = 0;
    for (std::size_t mesh_index = 0; mesh_index < total_triangle_meshes; mesh_index++) {

      const Object3D &current_triangle_mesh = mesh_bundle.getTriangleMesh(mesh_index);
      std::vector<aligned_vertex> vertices_array = align_vertices(current_triangle_mesh.vertices);
      const axstd::span<unsigned> &indices_array = current_triangle_mesh.indices;

      // Optix expects matrices to be represented in row major, so we use the cached transpose transform
      const glm::mat4 &transform_rm = mesh_bundle.reconstructTransform4x4(mesh_index).t;
      const float *flat_transform = glm::value_ptr(transform_rm);
      copy2device(vertices_array, indices_array, flat_transform, device_pointers_collection[mesh_index].geometry, device_allocations);
      OptixBuildInput input = create_trimesh_build_input(device_pointers_collection[mesh_index].geometry, vertices_array.size(), primitive_offset);
      primitive_offset += indices_array.size() / 3;
      inputs.push_back(input);
    }
    return inputs;
  }

  static OptixTraversableHandle compact_gas(OptixDeviceContext context,
                                            CUstream stream,
                                            OptixTraversableHandle handle,
                                            void *d_compact_size,
                                            std::size_t output_size_bytes,
                                            void **d_new_output_buffer) {
    std::size_t compacted_size{};
    DEVICE_ERROR_CHECK(cudaMemcpy(&compacted_size, d_compact_size, sizeof(std::size_t), cudaMemcpyDeviceToHost));
    if (compacted_size < output_size_bytes) {
      void *d_compact_output_buffer{};
      DEVICE_ERROR_CHECK(cudaMalloc(&d_compact_output_buffer, compacted_size));

      OptixTraversableHandle output_handle = 0;
      OPTIX_ERR_CHECK(
          optixAccelCompact(context, stream, handle, reinterpret_cast<CUdeviceptr>(d_compact_output_buffer), compacted_size, &output_handle));
      AX_ASSERT_NOTNULL(d_new_output_buffer);
      *d_new_output_buffer = d_compact_output_buffer;
      return output_handle;
    }
    return handle;
  }

  static OptixTraversableHandle build_triangle_gas(OptixDeviceContext context,
                                                   const primitive_aggregate_data_s &primitive_data_list,
                                                   void **d_outbuffer) {

    LOGS("Start building Optix GAS.");
    allocations_tracker_s allocations;
    std::vector<device_allocs_s> device_pointers;
    device_pointers.resize(primitive_data_list.mesh_geometry.triMeshCount());
    std::vector<OptixBuildInput> build_inputs = generate_trimesh_build_inputs(primitive_data_list, device_pointers, allocations);

    OptixAccelBuildOptions options = {};
#ifndef NDEBUG
    options.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_BUILD | OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
#else
    options.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE | OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
#endif
    options.operation = OPTIX_BUILD_OPERATION_BUILD;
    options.motionOptions.numKeys = 0;

    std::size_t *d_compact_size = nullptr;
    DEVICE_ERROR_CHECK(cudaMalloc(&d_compact_size, sizeof(std::size_t)));
    allocations.d_buffers.push_back(d_compact_size);
    OptixAccelEmitDesc property = {};
    property.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    property.result = reinterpret_cast<CUdeviceptr>(d_compact_size);

    OptixAccelBufferSizes buffer_sizes = {};
    OPTIX_ERR_CHECK(optixAccelComputeMemoryUsage(context, &options, build_inputs.data(), build_inputs.size(), &buffer_sizes));

    void *d_tmp = nullptr;
    DEVICE_ERROR_CHECK(cudaMalloc(&d_tmp, buffer_sizes.tempSizeInBytes));
    DEVICE_ERROR_CHECK(cudaMalloc(d_outbuffer, buffer_sizes.outputSizeInBytes));
    OptixTraversableHandle handle{};
    OPTIX_ERR_CHECK(optixAccelBuild(context,
                                    0,
                                    &options,
                                    build_inputs.data(),
                                    build_inputs.size(),
                                    reinterpret_cast<CUdeviceptr>(d_tmp),
                                    buffer_sizes.tempSizeInBytes,
                                    reinterpret_cast<CUdeviceptr>(*d_outbuffer),
                                    buffer_sizes.outputSizeInBytes,
                                    &handle,
                                    &property,
                                    1));

    DEVICE_ERROR_CHECK(cudaStreamSynchronize(0));
    void *d_compacted_output_buffer = nullptr;
    handle = compact_gas(context, 0, handle, d_compact_size, buffer_sizes.outputSizeInBytes, &d_compacted_output_buffer);
    std::swap(*d_outbuffer, d_compacted_output_buffer);
    DEVICE_ERROR_CHECK(cudaFree(d_compacted_output_buffer));

    DEVICE_ERROR_CHECK(cudaStreamSynchronize(0));

    AX_ASSERT(handle != 0, "Traversable handle is null!");
    allocations.d_buffers.push_back(d_tmp);
    device_dealloc(allocations);

    LOGS("Optix GAS built.");

    return handle;
  }
  extern "C" {
  extern const unsigned char PTX_EMBEDDED[];
  extern uint32_t PTX_EMBEDDEDLength;
  }
  void OptixAccelerator::build(primitive_aggregate_data_s primitive_data_list) {
    handle = build_triangle_gas(context, primitive_data_list, &d_outbuffer);

    OptixPipelineCompileOptions pipelineCompileOptions = create_pipeline_compile_options();
    OptixPipelineLinkOptions pipelineLinkOptions = create_pipeline_link_options();
    module = create_optix_module(context, &pipelineCompileOptions, reinterpret_cast<const char *>(PTX_EMBEDDED), PTX_EMBEDDEDLength, module_allocs);

    const char *raygen_ptx_entry = "__raygen__main";
    OptixProgramGroup raygen = create_raygen_pg(context, raygen_ptx_entry, module, pipeline_allocs);
    const char *miss_ptx_entry = "__miss__sample_envmap";
    OptixProgramGroup miss = create_miss_pg(context, miss_ptx_entry, module, pipeline_allocs);
    const char *any_ptx_entry = "__anyhit__random_intersect";
    const char *closest_ptx_entry = "__closesthit__minimum_intersect";
    OptixProgramGroup hitgroup = create_hit_pg(context, any_ptx_entry, closest_ptx_entry, module, module, pipeline_allocs);
    const char *exception_ptx_entry = "__exception__exception_handler";
    OptixProgramGroup exception = create_exception_pg(context, exception_ptx_entry, module, pipeline_allocs);

    programs[0] = raygen;
    programs[1] = miss;
    programs[2] = hitgroup;
    programs[3] = exception;
    pipeline = create_pipeline(context, programs, &pipelineCompileOptions, &pipelineLinkOptions, NUM_PROGRAMS, pipeline_allocs);

    raygen_record_s raygen_rcd{};
    misshit_record_s miss_rcd{};
    hit_record_s hit_rcd{};
    exception_record_s ex_rcd{};
    intersect_sbt = create_sbt(raygen, miss, hitgroup, exception, raygen_rcd, miss_rcd, hit_rcd, ex_rcd, sbt_allocs);

    DEVICE_ERROR_CHECK(cudaMalloc(&d_params_buffer, sizeof(optix_traversal_param_s)));
  }

  std::unique_ptr<DeviceIntersectorInterface> OptixAccelerator::getIntersectorObject() const {
    return std::make_unique<OptixIntersector>(handle, pipeline, nullptr, &intersect_sbt, reinterpret_cast<CUdeviceptr>(d_params_buffer));
  }

  void OptixAccelerator::copyParamsToDevice(const device_traversal_param_s &params) const {
    AX_ASSERT_NOTNULL(d_params_buffer);
    optix_traversal_param_s o_params;
    o_params.handle = handle;
    o_params.d_params = params;
    DEVICE_ERROR_CHECK(cudaMemcpy(reinterpret_cast<void *>(d_params_buffer), &o_params, sizeof(optix_traversal_param_s), cudaMemcpyHostToDevice));
  }

  void OptixAccelerator::cleanup() {
    if (pipeline) {
      OPTIX_ERR_CHECK(optixPipelineDestroy(pipeline));
      pipeline = {};
      LOG("Optix pipeline destroyed.", LogLevel::INFO);
    }
    if (module) {
      OPTIX_ERR_CHECK(optixModuleDestroy(module));
      module = {};
      LOG("Optix module destroyed.", LogLevel::INFO);
    }
    for (int i = 0; i < NUM_PROGRAMS; i++)
      if (programs[i]) {
        OPTIX_ERR_CHECK(optixProgramGroupDestroy(programs[i]));
        programs[i] = {};
        LOG("Optix program n°" + std::to_string(i) + " destroyed.", LogLevel::INFO);
      }

    if (d_outbuffer) {
      DEVICE_ERROR_CHECK(cudaFree(d_outbuffer));
      d_outbuffer = nullptr;
      LOG("Optix output device buffer freed.", LogLevel::INFO);
    }

    if (d_params_buffer) {
      DEVICE_ERROR_CHECK(cudaFree(reinterpret_cast<void *>(d_params_buffer)));
      LOG("Optix parameters device buffer freed.", LogLevel::INFO);
      d_params_buffer = nullptr;
    }

    LOG("Deallocating Optix shader table device buffers.", LogLevel::INFO);
    device_dealloc(sbt_allocs);
    LOG("Deallocating Optix pipeline device buffers.", LogLevel::INFO);
    device_dealloc(pipeline_allocs);
    LOG("Deallocating Optix module device buffers.", LogLevel::INFO);
    device_dealloc(module_allocs);

    LOG("Done cleaning up Optix structures.", LogLevel::INFO);
  }

}  // namespace nova::aggregate
