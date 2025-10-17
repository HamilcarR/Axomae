#include "aggregate/acceleration_interface.h"
#include "aggregate/aggregate_datastructures.h"
#include "api_common.h"
#include "api_datastructures.h"
#include "api_renderoptions.h"
#include "engine/nova_engine.h"
#include "manager/NovaExceptionManager.h"
#include "manager/NovaResourceManager.h"
#include "primitive/PrimitiveInterface.h"
#include "private_includes.h"
#include "shape/MeshContext.h"
#include "texturing/NovaTextureInterface.h"
#include "texturing/nova_texturing.h"
#include <internal/common/exception/GenericException.h>
#include <internal/common/math/math_utils.h>
#include <internal/common/utils.h>
#include <internal/debug/Logger.h>
#include <internal/geometry/Object3D.h>
#include <internal/macro/project_macros.h>
#include <internal/thread/worker/ThreadPool.h>
#include <memory>

namespace nova {
  class NvEngineInstance : public Engine {
    using RenderFutureResult = std::future<ERROR_STATE>;
    using RenderEngineInterfacePtr = std::unique_ptr<NovaRenderEngineInterface>;
    using EngineExceptionManagerPtr = std::unique_ptr<nova::NovaExceptionManager>;
    using ThreadpoolPtr = std::unique_ptr<threading::ThreadPool>;
    using AtomicIndex = std::atomic<unsigned>;

    static constexpr char NOVA_ASYNC_RENDER_TAG[] = "_NOVA_HOST_TAG";

    // TODO: properties need mutex protection or a move operation in rendering threads .

    NovaResourceManager manager;

    RenderEngineInterfacePtr render_engine;
    EngineExceptionManagerPtr engine_exception_manager;

    RenderBufferPtr render_buffer;

    RenderOptionsPtr render_options;  // is synced with the manager's data.
    EngineUserCallbackHandlerPtr user_callback;

    ScenePtr scene;
    RenderFutureResult render_result{};
    bool scene_built{false};
    gputils::gpu_util_structures_t gpu_structures;

    AtomicIndex frame_index{};

   public:
    static ThreadpoolPtr threadpool;

    NvEngineInstance() {
      render_options = create_renderoptions();
      render_buffer = create_renderbuffer();
      scene = create_scene();
      engine_exception_manager = std::make_unique<NovaExceptionManager>();
      render_engine = std::make_unique<NovaRenderEngineLR>();
      gpu_structures = nova::gputils::initialize_gpu_structures();
    }

    ~NvEngineInstance() override { nova::gputils::cleanup_gpu_structures(gpu_structures); }

    static ERROR_STATE synchronize() {
      if (!threadpool)
        return THREADPOOL_NOT_INITIALIZED;
      threadpool->fence(NOVA_ASYNC_RENDER_TAG);
      return SUCCESS;
    }

    static ERROR_STATE setThreadSize(unsigned threads) {
      try {
        synchronize();
        threadpool = std::make_unique<threading::ThreadPool>(threads);
      } catch (...) {
        return THREADPOOL_CREATION_ERROR;
      }
      return SUCCESS;
    }

    struct resource_holders_inits_s {
      std::size_t triangle_mesh_number;
      std::size_t triangle_number;

      std::size_t dielectrics_number;
      std::size_t diffuse_number;
      std::size_t conductors_number;

      std::size_t image_texture_number;
      std::size_t constant_texture_number;
      std::size_t hdr_texture_number;
    };

    static void initialize_resources_holders(nova::NovaResourceManager &manager, const resource_holders_inits_s &resrc) {
      nova::shape::shape_init_record_s shape_init_data{};
      shape_init_data.total_triangle_meshes = resrc.triangle_mesh_number;
      shape_init_data.total_triangles = resrc.triangle_number;
      auto &shape_reshdr = manager.getShapeData();
      shape_reshdr.init(shape_init_data);

      nova::primitive::primitive_init_record_s primitive_init_data{};
      primitive_init_data.geometric_primitive_count = resrc.triangle_number;
      primitive_init_data.total_primitive_count = primitive_init_data.geometric_primitive_count;
      auto &primitive_reshdr = manager.getPrimitiveData();
      primitive_reshdr.init(primitive_init_data);

      nova::material::material_init_record_s material_init_data{};
      material_init_data.conductors_size = resrc.conductors_number;
      material_init_data.dielectrics_size = resrc.dielectrics_number;
      material_init_data.diffuse_size = resrc.diffuse_number;
      auto &material_resrc = manager.getMaterialData();
      material_resrc.init(material_init_data);

      nova::texturing::texture_init_record_s texture_init_data{};
      texture_init_data.total_constant_textures = resrc.constant_texture_number;
      texture_init_data.total_image_textures = resrc.image_texture_number;
      auto &texture_resrc = manager.getTexturesData();
      texture_resrc.allocateMeshTextures(texture_init_data);
    }

    static shape::triangle::mesh_vbo_ids create_vbo_pack(const Trimesh &mesh) {
      shape::triangle::mesh_vbo_ids vbo_pack{};
      vbo_pack.vbo_positions = mesh.getInteropVertices();
      vbo_pack.vbo_indices = mesh.getInteropIndices();
      vbo_pack.vbo_normals = mesh.getInteropNormals();
      vbo_pack.vbo_uv = mesh.getInteropUVs();
      vbo_pack.vbo_tangents = mesh.getInteropTangents();
      return vbo_pack;
    }

    struct triangle_mesh_properties_t {
      std::size_t mesh_index;
      std::size_t triangle_index;
    };

    static glm::mat4 convert_transform(const float transform[16]) {
      glm::mat4 final_transform;
      for (int i = 0; i < 16; i++)
        glm::value_ptr(final_transform)[i] = transform[i];
      return final_transform;
    }

    static void add_trimesh_geometry(const Trimesh &mesh, const glm::mat4 &final_transform, NovaResourceManager &manager, bool use_interops) {
      nova::shape::ShapeResourcesHolder &res_holder = manager.getShapeData();
      const Object3D &geometry = to_obj3d(mesh);

      std::size_t stored_mesh_index = res_holder.addTriangleMesh(geometry);
      res_holder.addTransform(final_transform, stored_mesh_index);
      if (core::build::is_gpu_build && use_interops) {
        shape::triangle::mesh_vbo_ids vbo_pack = create_vbo_pack(mesh);
        res_holder.addTriangleMesh(vbo_pack);
      }
    }

    static void setup_trimesh(const Trimesh &mesh, const Material &material, NovaResourceManager &manager, bool uses_interops, size_t mesh_index) {
      const Object3D &geometry = to_obj3d(mesh);

      float transform_array[16] = {};
      const Transform &transform = mesh.getTransform();
      transform.getTransformMatrix(transform_array);
      const glm::mat4 transform_matrix = convert_transform(transform_array);

      add_trimesh_geometry(mesh, transform_matrix, manager, uses_interops);

      material::NovaMaterialInterface material_interface = setup_material_data(mesh, material, manager);
      for (std::size_t triangle_index = 0; triangle_index < geometry.indices.size(); triangle_index += Object3D::face_stride) {
        nova::shape::ShapeResourcesHolder &res_holder = manager.getShapeData();
        auto triangle = res_holder.addShape<nova::shape::Triangle>(mesh_index, triangle_index);
        manager.getPrimitiveData().addPrimitive<nova::primitive::NovaGeoPrimitive>(triangle, material_interface);
      }
    }

    static void setup_triangle_meshes(const Scene &scene, NovaResourceManager &manager, bool uses_interops) {
      std::size_t mesh_index = 0;
      for (size_t i = 0; i < scene.getMeshesNum(mesh::TRIANGLE); i++) {
        const Trimesh &geometry = *scene.getTriangleMeshCollection()[i];
        const Material &material = *scene.getMaterialCollection(mesh::TRIANGLE)[i];
        setup_trimesh(geometry, material, manager, uses_interops, mesh_index++);
      }
    }

    static void setup_meshes(const Scene &scene, NovaResourceManager &manager, bool uses_interops) {
      setup_triangle_meshes(scene, manager, uses_interops);
    }

    ERROR_STATE setScene(ScenePtr ptr) override {
      if (!ptr)
        return INVALID_ARGUMENT;
      scene = std::move(ptr);
      return SUCCESS;
    }

    const Scene *getScene() const override { return scene.get(); }

    Scene *getScene() override { return scene.get(); }

    ERROR_STATE buildScene() override {
      std::size_t number_primitives = scene->getPrimitivesNum(mesh::TRIANGLE);
      std::size_t number_trimesh = scene->getMeshesNum(mesh::TRIANGLE);
      std::size_t total_mesh_number = number_trimesh;

      resource_holders_inits_s resrc{};
      resrc.triangle_mesh_number = total_mesh_number;
      resrc.triangle_number = number_primitives;
      resrc.conductors_number = total_mesh_number;
      resrc.diffuse_number = total_mesh_number;
      resrc.dielectrics_number = total_mesh_number;
      resrc.image_texture_number = total_mesh_number * PBR_TEXTURE_PACK_SIZE;
      resrc.hdr_texture_number = 1;  // At least 1 for Environment map.
      initialize_resources_holders(manager, resrc);
      setup_meshes(*scene, manager, render_options->isUsingInterops());

      scene_built = true;
      return SUCCESS;
    }

    static aggregate::DefaultAccelerator build_cpu_managed_accelerator(const aggregate::primitive_aggregate_data_s &primitive_geometry) {
      aggregate::DefaultAccelerator accelerator;
      accelerator.build(primitive_geometry);
      return accelerator;
    }

    static std::unique_ptr<aggregate::DeviceAcceleratorInterface> build_device_accelerator(
        const aggregate::primitive_aggregate_data_s &primitive_geometry) {
      auto device_builder = aggregate::DeviceAcceleratorInterface::make();
      device_builder->build(primitive_geometry);
      return device_builder;
    }

    ERROR_STATE buildAcceleration() override {
      if (!scene_built)
        return SCENE_NOT_PROCESSED;
      primitive::primitives_view_tn primitive_list_view = manager.getPrimitiveData().getPrimitiveView();
      shape::MeshBundleViews mesh_geometry = manager.getShapeData().getMeshSharedViews();
      aggregate::primitive_aggregate_data_s aggregate;
      aggregate.primitive_list_view = primitive_list_view;
      aggregate.mesh_geometry = mesh_geometry;

      auto accelerator = build_cpu_managed_accelerator(aggregate);
      manager.setManagedCpuAccelerationStructure(std::move(accelerator));

      if (core::build::is_gpu_build) {
        auto device_accelerator = build_device_accelerator(aggregate);
        manager.setManagedGpuAccelerationStructure(std::move(device_accelerator));
      }

      return SUCCESS;
    }

    ERROR_STATE cleanup() override {
      manager.clearResources();
      manager = {};
      scene_built = false;
      if (!render_buffer || !scene)
        return INVALID_ENGINE_STATE;
      render_buffer->resetBuffers();
      scene->cleanup();
      return SUCCESS;
    }

    ERROR_STATE setRenderBuffers(RenderBufferPtr buffers) override {
      if (!buffers)
        return INVALID_BUFFER_STATE;
      render_buffer = std::move(buffers);
      return SUCCESS;
    }

    static void allocate_environment_maps(texturing::TextureResourcesHolder &resrc, const nova::Scene &scene) {
      CsteTextureCollection env_collection = scene.getEnvmapCollection();
      resrc.allocateEnvironmentMaps(env_collection.size());
      int env_id = scene.getCurrentEnvmapId();
      AX_ASSERT_GE(env_id, 0);
      resrc.setEnvmapId(env_id);
      for (const TexturePtr &texture : env_collection) {
        AX_ASSERT_EQ(texture->getFormat(), texture::FLOATX4);
        std::size_t index = resrc.addTexture(static_cast<const float *>(texture->getTextureBuffer()),
                                             texture->getWidth(),
                                             texture->getHeight(),
                                             texture->getChannels(),
                                             texture->getInvertX(),
                                             texture->getInvertY(),
                                             texture->getInteropID());
        resrc.addNovaTexture<texturing::EnvmapTexture>(index);
      }
    }

    bool isRendering() const override { return manager.getEngineData().is_rendering; }

    /* returns true if exception needs to shutdown renderer  */
    bool isExceptionShutdown() const {
      using namespace exception;
      if (get_error_status(*engine_exception_manager) != 0) {
        auto exception_list = get_error_list(*engine_exception_manager);
        bool must_shutdown = false;
        for (const auto &elem : exception_list) {
          switch (elem) {
            case NOERR:
              must_shutdown |= false;
              break;
            case INVALID_RENDER_MODE:
              LOGS("Render mode not selected.");
              must_shutdown = true;
              break;
            case INVALID_INTEGRATOR:
              LOGS("Provided integrator is invalid.");
              must_shutdown = true;
              break;
            case INVALID_SAMPLER_DIM:
            case SAMPLER_INIT_ERROR:
            case SAMPLER_DOMAIN_EXHAUSTED:
            case SAMPLER_INVALID_ARG:
            case SAMPLER_INVALID_ALLOC:
              LOGS("Sampler initialization error");
              must_shutdown = true;
              break;
            case GENERAL_ERROR:
              LOGS("Renderer general error.");
              must_shutdown = true;
              break;
            case INVALID_RENDERBUFFER_STATE:
              LOGS("Render buffer is invalid.");
              must_shutdown = true;
              break;
            case INVALID_ENGINE_INSTANCE:
              LOGS("Engine instance is invalid.");
              must_shutdown = true;
              break;
            case INVALID_RENDERBUFFER_DIM:
              LOGS("Wrong buffer width / height dimension.");
              must_shutdown = true;
              break;
            default:
              must_shutdown |= false;
              break;
          }
        }
        return must_shutdown;
      }
      return false;
    }

    unsigned getFrameIndex() const override { return frame_index; }

    void getRenderEngineError(char error_log[1024], size_t &size) override {}

    void writeFramebuffer() {
      HdrBufferStruct rb = render_buffer->getRenderBuffers();
      FloatView accumulator_buffer = render_buffer->getAccumulator();
      FloatView final_buffer = render_buffer->backBuffer();

      AX_ASSERT_FALSE(accumulator_buffer.empty());
      for (size_t i = 0; i < render_buffer->getHeight() * render_buffer->getWidth() * render_buffer->getChannel(texture::COLOR); i++) {
        float partial = rb.partial_buffer[i];
        accumulator_buffer[i] += partial;
        float accumulated = accumulator_buffer[i] / float(frame_index + 1);
        final_buffer[i] = accumulated;
      }
    }

    struct render_data_s {
      HdrBufferStruct buffers;
      unsigned image_width{}, image_height{};
    };

    class EngineCallbackManager {
      EngineUserCallbackHandler *cback;

     public:
      EngineCallbackManager(EngineUserCallbackHandler *opt) {
        cback = opt;
        if (cback)
          cback->framePreRenderRoutine();
      }
      ~EngineCallbackManager() {
        if (cback)
          cback->framePostRenderRoutine();
      }

      EngineCallbackManager(const EngineCallbackManager &) = default;
      EngineCallbackManager(EngineCallbackManager &&) = delete;
      EngineCallbackManager &operator=(const EngineCallbackManager &) = default;
      EngineCallbackManager &operator=(EngineCallbackManager &&) = delete;
    };

    static void setup_camera(NovaResourceManager &manager, Camera *scene_camera) {
      camera::CameraResourcesHolder &camera = manager.getCameraData();
      if (!scene_camera)
        return;

      float vec3[3]{}, mat16[16]{};

      scene_camera->getUpVector(vec3);
      camera.up_vector = f3_to_vec3(vec3);

      scene_camera->getProjectionMatrix(mat16);
      camera.P = f16_to_mat4(mat16);

      scene_camera->getViewMatrix(mat16);
      camera.V = f16_to_mat4(mat16);

      scene_camera->getProjectionViewMatrix(mat16);
      camera.PV = f16_to_mat4(mat16);

      scene_camera->getInverseProjectionViewMatrix(mat16);
      camera.inv_PV = f16_to_mat4(mat16);

      scene_camera->getInverseProjectionMatrix(mat16);
      camera.inv_P = f16_to_mat4(mat16);

      scene_camera->getInverseViewMatrix(mat16);
      camera.inv_V = f16_to_mat4(mat16);

      scene_camera->getPosition(vec3);
      camera.position = f3_to_vec3(vec3);

      scene_camera->getDirection(vec3);
      camera.direction = f3_to_vec3(vec3);

      camera.far = scene_camera->getClipPlaneFar();
      camera.near = scene_camera->getClipPlaneNear();
      camera.fov = scene_camera->getFov();
      camera.screen_width = scene_camera->getResolutionWidth();
      camera.screen_height = scene_camera->getResolutionHeight();
    }

    static nova::device_traversal_param_s fill_params(const nova::HdrBufferStruct &render_buffers,
                                                      unsigned grid_width,
                                                      unsigned grid_height,
                                                      unsigned grid_depth,
                                                      unsigned sample_index,
                                                      const nova::NovaResourceManager *resrc_manager,
                                                      const nova::gputils::gpu_util_structures_t &gpu_utils) {

      nova::device_traversal_param_s params;
      params.device_random_generators.rqmc_generator = gpu_utils.random_generator.sobol;
      params.render_buffers = render_buffers;
      params.material_view = resrc_manager->getMaterialData().getMaterialView();
      params.mesh_bundle_views = resrc_manager->getShapeData().getMeshSharedViews();
      params.primitives_view = resrc_manager->getPrimitiveData().getPrimitiveView();
      params.texture_bundle_views = resrc_manager->getTexturesData().getTextureBundleViews();
      params.current_envmap_index = resrc_manager->getTexturesData().getEnvmapId();
      params.camera = resrc_manager->getCameraData();
      params.width = grid_width;
      params.height = grid_height;
      params.depth = grid_depth;

      params.sample_max = resrc_manager->getEngineData().renderer_max_samples;
      params.sample_index = sample_index;
      return params;
    }

    ERROR_STATE renderGpu(render_data_s render_data) {
      unsigned current_cam_id = scene->getCurrentCameraId();
      Camera *scene_camera = scene->getCamera(current_cam_id);
      setup_camera(manager, scene_camera);
      while (frame_index < render_options->getMaxSamples() && isRendering()) {
        int new_depth = manager.getEngineData().max_depth < render_options->getMaxDepth() ? manager.getEngineData().max_depth + 1 :
                                                                                            render_options->getMaxDepth();
        manager.getEngineData().sample_increment = frame_index;
        manager.getEngineData().max_depth = new_depth;
        manager.getEngineData().renderer_max_samples = render_options->getMaxSamples();
        EngineCallbackManager callback_manager(user_callback.get());

        nova::device_traversal_param_s traversal_parameters = fill_params(render_buffer->getRenderBuffers(),
                                                                          render_buffer->getWidth(),
                                                                          render_buffer->getHeight(),
                                                                          new_depth,
                                                                          frame_index,
                                                                          &manager,
                                                                          gpu_structures);

        manager.registerDeviceParameters(traversal_parameters);
        try {
          nova::nova_eng_internals internals;
          internals.resource_manager = &manager;
          internals.exception_manager = engine_exception_manager.get();
          nova::gpu_draw(traversal_parameters, internals);
        } catch (const ::exception::GenericException &e) {
          DEVICE_ERROR_CHECK(device::gpgpu::synchronize_device().error_status);
          return RENDERER_EXCEPTION;
        }

        if (isExceptionShutdown())
          return RENDERER_EXCEPTION;

        writeFramebuffer();
        render_buffer->swapBackBuffer();
        frame_index++;
      }
      return SUCCESS;
    }

    ERROR_STATE renderCpu(render_data_s render_data) {
      unsigned current_cam_id = scene->getCurrentCameraId();
      Camera *scene_camera = scene->getCamera(current_cam_id);
      setup_camera(manager, scene_camera);
      float serie_max = math::calculus::compute_serie_term(render_options->getMaxSamples());
      while (frame_index < render_options->getMaxSamples() && isRendering()) {
        manager.getEngineData().sample_increment = frame_index;
        int new_depth = manager.getEngineData().max_depth < render_options->getMaxDepth() ? manager.getEngineData().max_depth + 1 :
                                                                                            render_options->getMaxDepth();
        manager.getEngineData().max_depth = new_depth;
        EngineCallbackManager callback_manager(user_callback.get());
        try {
          nova_eng_internals internals{&manager, engine_exception_manager.get()};
          draw(&render_data.buffers, render_data.image_width, render_data.image_height, render_engine.get(), threadpool.get(), internals);
          synchronize();
        } catch (const ::exception::GenericException &e) {
          synchronize();
          return RENDERER_EXCEPTION;
        }
        if (isExceptionShutdown())
          return RENDERER_EXCEPTION;
        writeFramebuffer();

        render_buffer->swapBackBuffer();
        frame_index++;
      }
      return SUCCESS;
    }

    ERROR_STATE preRenderChecks() const {
      if (!threadpool)
        return THREADPOOL_NOT_INITIALIZED;
      if (!render_options)
        return RENDER_OPTIONS_NOT_INITIALIZED;
      if (!render_buffer)
        return RENDER_BUFFER_NOT_INITIALIZED;
      if (!render_engine)
        return RENDER_ENGINE_NOT_INITIALIZED;
      if (!scene_built)
        return SCENE_NOT_PROCESSED;

      return SUCCESS;
    }

    static void setup_engine_data(engine::EngineResourcesHolder &resrc, const RenderOptions &render_options) {
      resrc.aliasing_samples = render_options.getAliasingSamples();
      resrc.max_depth = render_options.getMaxDepth();
      resrc.sample_increment = render_options.getSamplesIncrement();
      resrc.vertical_invert = render_options.isFlippedV();
      resrc.tiles_width = render_options.getTileDimensionWidth();
      resrc.tiles_height = render_options.getTileDimensionHeight();
      resrc.integrator_flag = render_options.getIntegratorFlag();
      resrc.threadpool_tag = NOVA_ASYNC_RENDER_TAG;
    }

    using RenderCallback = std::function<ERROR_STATE(render_data_s)>;

    ERROR_STATE startRender() override {
      HdrBufferStruct buffers = render_buffer->getRenderBuffers();
      ERROR_STATE pre_render_checks = preRenderChecks();
      if (pre_render_checks != SUCCESS)
        return pre_render_checks;
      RenderCallback render_callback;
      render_data_s rd;
      rd.buffers = buffers;
      rd.image_width = render_buffer->getWidth();
      rd.image_height = render_buffer->getHeight();
      if (!render_options->isUsingGpu()) {
        render_callback = [&](render_data_s data) { return renderCpu(data); };
      } else {
        render_callback = [&](render_data_s data) { return renderGpu(data); };
      }

      manager.getEngineData().is_rendering = true;
      render_result = threadpool->addTask(threading::ALL_TASK, render_callback, rd);
      return SUCCESS;
    }

    ERROR_STATE prepareRender() override {
      texturing::TextureResourcesHolder &textures_manager = manager.getTexturesData();
      shape::ShapeResourcesHolder &shape_manager = manager.getShapeData();
      engine::EngineResourcesHolder &opts_manager = manager.getEngineData();
      setup_engine_data(opts_manager, *render_options);
      allocate_environment_maps(textures_manager, *scene);
      if (render_options->isUsingGpu()) {
        shape_manager.lockResources();
        shape_manager.mapBuffers();
        textures_manager.lockResources();
        textures_manager.mapBuffers();
      }
      return SUCCESS;
    }

    static ERROR_STATE syncAndEmptyPool() {
      if (!threadpool)
        return THREADPOOL_NOT_INITIALIZED;
      /* Empty scheduler list. */
      threadpool->emptyQueue(NOVA_ASYNC_RENDER_TAG);
      /* Synchronize the threads. */
      threadpool->fence(NOVA_ASYNC_RENDER_TAG);
      threadpool->fence(threading::ALL_TASK);
      return SUCCESS;
    }

    ERROR_STATE stopRender() override {
      manager.getEngineData().is_rendering = false;
      ERROR_STATE err = syncAndEmptyPool();
      /* unmap gpu resources */
      manager.getShapeData().releaseResources();
      manager.getTexturesData().releaseResources();
      frame_index = 1;
      if (err != SUCCESS)
        return err;
      return SUCCESS;
    }

    ERROR_STATE setRenderOptions(RenderOptionsPtr opts) override {
      if (!opts)
        return INVALID_ARGUMENT;
      render_options = std::move(opts);
      return SUCCESS;
    }

    void setUserCallback(EngineUserCallbackHandlerPtr callback) override { user_callback = std::move(callback); }

    const RenderOptions *getRenderOptions() const override { return render_options.get(); }

    RenderOptions *getRenderOptions() override { return render_options.get(); }

    const RenderBuffer *getRenderBuffers() const override { return render_buffer.get(); }

    RenderBuffer *getRenderBuffers() override { return render_buffer.get(); }

    NovaResourceManager &getResrcManager() override { return manager; }
  };

  std::unique_ptr<threading::ThreadPool> NvEngineInstance::threadpool = nullptr;
  ERROR_STATE init_threads(unsigned number_threads) { return NvEngineInstance::setThreadSize(number_threads); }

  EnginePtr create_engine() { return std::make_unique<NvEngineInstance>(); }

}  // namespace nova
