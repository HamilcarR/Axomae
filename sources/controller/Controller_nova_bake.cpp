#include "Config.h"
#include "EnvmapTextureManager.h"
#include "ExceptionHandlerUI.h"
#include "GUIWindow.h"
#include "TextureViewerWidget.h"
#include "WorkspaceTracker.h"
#include "engine/nova_exception.h"
#include "integrator/Integrator.h"
#include "internal/macro/project_macros.h"
#include "manager/ManagerInternalStructs.h"
#include "manager/NovaResourceManager.h"
#include "nova/bake.h"
#include "nova/bake_render_data.h"
#include <QFileDialog>
#include <future>
#include <internal/common/exception/GenericException.h>
#include <internal/debug/Logger.h>
#include <internal/macro/exception_macros.h>

namespace exception {
  GENERIC_EXCEPT_DEFINE(InValidCameraException, "No valid camera present in the scene , or the camera hasn't been initialized.", WARNING)
  GENERIC_EXCEPT_DEFINE(InValidRendererException, "The rasterizer hasn't been intialized.", WARNING)
  GENERIC_EXCEPT_DEFINE(InvalidSceneException, "Current scene is not initialized.", WARNING)
  GENERIC_EXCEPT_DEFINE(InvalidInputException, "Invalid input.", INFO)
  GENERIC_EXCEPT_DEFINE(InvalidRenderBufferStateException, "Render buffers are in an invalid state.", CRITICAL)
  GENERIC_EXCEPT_DEFINE(InvalidSceneNodeTypeException, "Scene node type discrepancy.", CRITICAL)
  GENERIC_EXCEPT_DEFINE(NullTextureWidgetException, "Failed to create output texture.", CRITICAL)
  const char *const CRITICAL_TXT = "A critical error occured. \nSee the logs for more precision.";

}  // namespace exception

const char *const NOVABAKE_POOL_TAG = "_NOVABAKE_POOL_TAG_";
const char *const NOVABAKE_GPU_POOL_TAG = "NOVABAKE_GPU_POOL_TAG_";
/**************************************************************************************************************/
namespace controller {
  /* To read options from the UI and transmit them to the bakery*/
  struct ui_render_options {
    int aa_samples;
    int max_samples;
    int max_depth;
    int tiles;
    int width;
    int height;
  };
  struct engine_misc_options {
    int engine_type_flag;
    bool flip_v;
    bool use_gpu;
  };

  static void setup_camera(nova_baker_utils::engine_data &engine_data, const Camera &renderer_camera, int render_width, int render_height) {
    /* Camera */
    nova_baker_utils::camera_data camera_data{};
    const float fov_deg = renderer_camera.getFov();
    const float near = renderer_camera.getNear();
    const float far = renderer_camera.getFar();
    const float ratio = (float)render_width / (float)render_height;
    const glm::mat4 new_projection = Camera::computeProjectionMatrix(fov_deg, near, far, ratio);

    camera_data.position = renderer_camera.getPosition();
    camera_data.direction = renderer_camera.getDirection();
    camera_data.up_vector = renderer_camera.getUpVector();
    camera_data.projection = new_projection;
    camera_data.view = renderer_camera.getTransformedView();
    camera_data.far = far;
    camera_data.near = near;
    camera_data.fov = glm::radians(fov_deg);

    engine_data.camera = camera_data;
  }

  static void setup_scene_transformation(nova_baker_utils::engine_data &engine_data, const Camera &renderer_camera) {
    /* Scene transformations */
    nova_baker_utils::scene_transform_data scene_transfo{};
    scene_transfo.root_rotation = renderer_camera.getSceneRotationMatrix();
    scene_transfo.root_transformation = renderer_camera.getLocalModelMatrix();
    scene_transfo.root_translation = renderer_camera.getSceneTranslationMatrix();

    engine_data.scene = scene_transfo;
  }

  static void setup_engine(nova_baker_utils::engine_data &engine_data,
                           const ui_render_options &engine_options,
                           const engine_misc_options &misc_options) {
    engine_data.aa_samples = engine_options.aa_samples;
    engine_data.depth_max = engine_options.max_depth;
    engine_data.samples_max = engine_options.max_samples;
    engine_data.num_tiles_w = engine_data.num_tiles_h = engine_options.tiles;
    engine_data.samples_increment = engine_data.samples_max;
    engine_data.engine_type_flag = misc_options.engine_type_flag;
    engine_data.flip_v = misc_options.flip_v;
    engine_data.threadpool_tag = NOVABAKE_POOL_TAG;
  }

  /* Retrieve all envmaps registered.*/
  static void setup_envmaps(const EnvmapTextureManager &envmap_manager, nova_baker_utils::engine_data &engine_data) {
    axstd::span<const texture::envmap::EnvmapTextureGroup> baked_envmaps = envmap_manager.getBakesViews();
    std::vector<nova_baker_utils::envmap_data_s> envmap_data_collection;
    for (const auto &envmap : baked_envmaps) {
      nova_baker_utils::envmap_data_s data{};
      data.equirect_glID = envmap.equirect_id;
      AX_ASSERT_NOTNULL(envmap.metadata);
      data.width = envmap.metadata->metadata.width;
      data.height = envmap.metadata->metadata.height;
      data.raw_data = envmap.metadata->data.data();
      data.channels = envmap.metadata->metadata.channels;
      envmap_data_collection.push_back(data);
    }
    engine_data.environment_maps = envmap_data_collection;
  }

  nova_baker_utils::engine_data generate_engine_data(const std::vector<Mesh *> &mesh_collection,
                                                     const ui_render_options &render_options,
                                                     const engine_misc_options &misc_options,
                                                     const Camera &renderer_camera,
                                                     const RendererInterface &renderer) {
    nova_baker_utils::engine_data engine_data{};
    engine_data.mesh_list = &mesh_collection;
    setup_engine(engine_data, render_options, misc_options);
    setup_camera(engine_data, renderer_camera, render_options.width, render_options.height);
    setup_scene_transformation(engine_data, renderer_camera);
    setup_envmaps(renderer.getCurrentEnvmapId(), engine_data);
    return engine_data;
  }

  std::unique_ptr<nova::HdrBufferStruct> allocate_buffers(nova_baker_utils::NovaBakingStructure &nova_baking_structure,
                                                          const ui_render_options &render_options) {

    /* buffers*/
    image::ThumbnailImageHolder<float> &image_holder = nova_baking_structure.bake_buffers.image_holder;
    auto &partial = nova_baking_structure.bake_buffers.partial;
    auto &accumulator = nova_baking_structure.bake_buffers.accumulator;
    auto &depth = nova_baking_structure.bake_buffers.depth;
    const int COLOR_CHANS = 4;
    const int DEPTH_CHANS = 2;
    size_t color_buffer_size = render_options.width * render_options.height * COLOR_CHANS;
    size_t depth_buffer_size = render_options.width * render_options.height * DEPTH_CHANS;
    image_holder.data.resize(color_buffer_size);
    partial.resize(color_buffer_size);
    accumulator.resize(color_buffer_size);
    depth.reserve(depth_buffer_size);  // only 2 channel needed one for min distance , one for max , initialized at max/min distance.
    for (int i = 0; i < depth_buffer_size; i += 2) {
      depth.push_back(1e30f);
      depth.push_back(-1e30f);
    }

    image_holder.metadata.channels = COLOR_CHANS;
    image_holder.metadata.color_corrected = false;
    image_holder.metadata.format = "hdr";
    image_holder.metadata.height = render_options.height;
    image_holder.metadata.width = render_options.width;
    image_holder.metadata.is_hdr = true;
    std::unique_ptr<nova::HdrBufferStruct> buffers = std::make_unique<nova::HdrBufferStruct>();

    buffers->partial_buffer = partial.data();
    buffers->accumulator_buffer = accumulator.data();
    buffers->depth_buffer = depth.data();
    buffers->byte_size_color_buffers = color_buffer_size * sizeof(float);
    buffers->byte_size_depth_buffers = depth_buffer_size * sizeof(float);
    buffers->channels = COLOR_CHANS;
    return buffers;
  }

  static void setup_render_scene_data(const ui_render_options &render_options,
                                      nova::NovaResourceManager *manager,
                                      std::unique_ptr<nova::NovaExceptionManager> exception_manager,
                                      std::unique_ptr<NovaRenderEngineInterface> engine_instance,
                                      const nova::device_shared_caches_t &shared_caches,
                                      nova_baker_utils::NovaBakingStructure &nova_baking_structure,
                                      threading::ThreadPool *thread_pool) {
    /* Engine type */
    nova_baking_structure.render_context = nova_baker_utils::render_scene_context{};
    nova_baker_utils::render_scene_context &render_scene_data = nova_baking_structure.render_context;
    render_scene_data.buffers = allocate_buffers(nova_baking_structure, render_options);
    render_scene_data.width = render_options.width;
    render_scene_data.height = render_options.height;
    render_scene_data.engine_instance = std::move(engine_instance);
    render_scene_data.nova_resource_manager = manager;
    render_scene_data.nova_exception_manager = std::move(exception_manager);
    render_scene_data.thread_pool = thread_pool;
    render_scene_data.shared_caches = shared_caches;
  }

  /**************************************************************************************************************/

  static std::unique_ptr<HdrRenderViewerWidget> texture_display_render(Controller *app_controller, const image::ThumbnailImageHolder<float> &img) {
    return std::make_unique<HdrRenderViewerWidget>(&img, app_controller);
  }

  /* returns true if exception needs to shutdown renderer  */
  static bool on_exception_shutdown(const nova::NovaExceptionManager &exception_manager) {
    using namespace nova::exception;
    if (nova_baker_utils::get_error_status(exception_manager) != 0) {
      auto exception_list = nova_baker_utils::get_error_list(exception_manager);
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

  static void color_correct_buffers(const nova::HdrBufferStruct *buffers, image::ImageHolder<float> &image_holder, float current_sample_num) {
    if (!buffers) {
      LOG("Render buffers are in an invalid state.", LogLevel::CRITICAL);
      return;
    }
    for (int i = 0; i < image_holder.metadata.height * image_holder.metadata.width * image_holder.metadata.channels; i++) {
      float accumulated = buffers->accumulator_buffer[i] / (current_sample_num + 1);
      float partial = buffers->partial_buffer[i];
      float interpolated = accumulated + 0.5f * (partial - accumulated);
      image_holder.data[i] = interpolated;
    }
  }

  struct progressive_render_metadata {
    int max_depth;
    int max_samples;
    int serie_max;
    bool is_rendering;
  };

  static progressive_render_metadata create_render_metadata(const nova_baker_utils::render_scene_context &render_scene_data) {
    int max_depth = render_scene_data.nova_resource_manager->getEngineData().max_depth;
    int max_samples = render_scene_data.nova_resource_manager->getEngineData().renderer_max_samples;
    int smax = math::calculus::compute_serie_term(max_samples);
    bool is_rendering = render_scene_data.nova_resource_manager->getEngineData().is_rendering;
    return {max_depth, max_samples, smax, is_rendering};
  }

  static bool is_rendering(nova_baker_utils::render_scene_context &render_context) {
    return render_context.nova_resource_manager->getEngineData().is_rendering;
  }

  static void display_render(controller::Controller *app_controller, const image::ThumbnailImageHolder<float> &image_holder) {
    /* Creates a texture widget that will display the rendered image */
    nova_baker_utils::NovaBakingStructure &nova_baking_structure = app_controller->getBakingStructure();
    auto &texture_widget = nova_baking_structure.spawned_window;
    texture_widget = texture_display_render(app_controller, image_holder);
    if (!texture_widget)
      throw exception::NullTextureWidgetException();
    QWidget *img_ptr = texture_widget.get();
    img_ptr->show();
  }

  class CPUWorkerRetAction : public nova_baker_utils::WorkerRetAction {
   public:
    CPUWorkerRetAction() = default;
    void cleanup() override { LOG("Finishing CPU Worker", LogLevel::INFO); }
  };

  static std::unique_ptr<nova_baker_utils::WorkerRetAction> do_progressive_render(nova_baker_utils::render_scene_context &render_scene_data,
                                                                                  image::ImageHolder<float> &image_holder) {
    int sample_increment = 1;
    progressive_render_metadata metadata = create_render_metadata(render_scene_data);
    auto action = std::make_unique<CPUWorkerRetAction>();
    while (sample_increment < metadata.serie_max && is_rendering(render_scene_data)) {
      render_scene_data.nova_resource_manager->getEngineData().sample_increment = sample_increment;
      int new_depth = render_scene_data.nova_resource_manager->getEngineData().max_depth < metadata.max_depth ?
                          render_scene_data.nova_resource_manager->getEngineData().max_depth + 1 :
                          metadata.max_depth;
      render_scene_data.nova_resource_manager->getEngineData().max_depth = new_depth;

      try {
        nova_baker_utils::bake_scene(render_scene_data);
      } catch (const exception::GenericException &e) {
        ExceptionInfoBoxHandler::handle(e);
        nova_baker_utils::synchronize_render_threads(render_scene_data, NOVABAKE_POOL_TAG);
        return action;
      }
      nova_baker_utils::synchronize_render_threads(render_scene_data, NOVABAKE_POOL_TAG);

      if (on_exception_shutdown(*render_scene_data.nova_exception_manager))
        return action;
      color_correct_buffers(render_scene_data.buffers.get(), image_holder, (float)sample_increment);
      sample_increment++;
    }
    return action;
  }

#ifdef AXOMAE_USE_CUDA
  template<class T>
  class GPUWorkerRetAction : public nova_baker_utils::WorkerRetAction {
    T gpu_context;

   public:
    explicit GPUWorkerRetAction(T &&ctx) : gpu_context(std::move(ctx)) {}
    void pushContext() { device::gpgpu::set_current_context(gpu_context); }
    void popContext() { device::gpgpu::get_current_context(gpu_context); }
    void cleanup() override { LOG("Finishing GPU Worker", LogLevel::INFO); }
  };

  static nova::device_traversal_param_s fill_params(const nova::HdrBufferStruct &render_buffers,
                                                    unsigned grid_width,
                                                    unsigned grid_height,
                                                    unsigned grid_depth,
                                                    const nova::NovaResourceManager *resrc_manager,
                                                    const nova::gputils::gpu_util_structures_t &gpu_utils) {

    nova::device_traversal_param_s params;
    params.device_random_generators.sobol = gpu_utils.random_generator.sobol;
    params.render_buffers = render_buffers;
    params.material_view = resrc_manager->getMaterialData().getMaterialView();
    params.mesh_bundle_views = resrc_manager->getShapeData().getMeshSharedViews();
    params.primitives_view = resrc_manager->getPrimitiveData().getPrimitiveView();
    params.texture_bundle_views = resrc_manager->getTexturesData().getTextureBundleViews();
    params.environment_map = resrc_manager->getTexturesData().getEnvmap();
    params.camera = resrc_manager->getCameraData();
    params.width = grid_width;
    params.height = grid_height;
    params.depth = grid_depth;

    return params;
  }

  static std::unique_ptr<nova_baker_utils::WorkerRetAction> do_progressive_render_gpu(nova_baker_utils::render_scene_context &render_scene_data,
                                                                                      image::ImageHolder<float> &image_holder,
                                                                                      device::gpgpu::GPUContext &&gpu_context) {

    nova::gputils::domain2d domain{(unsigned)render_scene_data.width, (unsigned)render_scene_data.height};
    nova::gputils::initialize_gpu_structures(domain, render_scene_data.gpu_util_structures);

    nova::device_shared_caches_t &buffer_collection = render_scene_data.shared_caches;
    nova::gputils::lock_host_memory_default(buffer_collection);

    progressive_render_metadata metadata = create_render_metadata(render_scene_data);

    nova::device_traversal_param_s traversal_parameters = fill_params(*render_scene_data.buffers,
                                                                      render_scene_data.width,
                                                                      render_scene_data.height,
                                                                      1,
                                                                      render_scene_data.nova_resource_manager,
                                                                      render_scene_data.gpu_util_structures);

    render_scene_data.nova_resource_manager->registerDeviceParameters(traversal_parameters);

    auto act = std::make_unique<GPUWorkerRetAction<device::gpgpu::GPUContext>>(std::move(gpu_context));
    act->pushContext();

    int sample_increment = 1;
    //    while (sample_increment < metadata.serie_max && is_rendering(render_scene_data)) {
    render_scene_data.nova_resource_manager->getEngineData().sample_increment = sample_increment;
    int new_depth = render_scene_data.nova_resource_manager->getEngineData().max_depth < metadata.max_depth ?
                        render_scene_data.nova_resource_manager->getEngineData().max_depth + 1 :
                        metadata.max_depth;
    render_scene_data.nova_resource_manager->getEngineData().max_depth = new_depth;
    try {
      nova::nova_eng_internals internals;
      internals.resource_manager = render_scene_data.nova_resource_manager;
      internals.exception_manager = render_scene_data.nova_exception_manager.get();
      nova::gpu_draw(traversal_parameters, internals);
    } catch (const exception::GenericException &e) {
      ExceptionInfoBoxHandler::handle(e);
      nova::gputils::unlock_host_memory(buffer_collection);
      act->popContext();
      return act;
    }

    if (on_exception_shutdown(*render_scene_data.nova_exception_manager)) {
      nova::gputils::unlock_host_memory(buffer_collection);
      act->popContext();
      return act;
    }
    color_correct_buffers(render_scene_data.buffers.get(), image_holder, (float)sample_increment);
    sample_increment++;
    //  }
    nova::gputils::unlock_host_memory(buffer_collection);
    act->popContext();
    return act;
  }

  static void map_resources(nova_baker_utils::render_scene_context &render_scene_data) {
    nova::shape::ShapeResourcesHolder &shape_manager = render_scene_data.nova_resource_manager->getShapeData();
    shape_manager.lockResources();
    shape_manager.mapBuffers();
    nova::texturing::TextureResourcesHolder &textures_manager = render_scene_data.nova_resource_manager->getTexturesData();
    textures_manager.lockResources();
    textures_manager.mapBuffers();
  }
#endif

  static void start_baking(nova_baker_utils::render_scene_context &render_scene_data,
                           Controller *app_controller,
                           image::ThumbnailImageHolder<float> &image_holder,
                           bool use_gpu) {
    try {
      display_render(app_controller, image_holder);
    } catch (const std::exception &e) {
      throw;
    }
    /*
     * Using cuda GL interop necessitates the acquisition of resources to happen in the main thread because of the GL context.
     * so we need to initialize graphics resource handles before launching the baking worker thread , and release it in NovaStopBake().
     */
    DisplayManager3D &disp_manager = app_controller->getDisplayManager();
    nova_baker_utils::NovaBakingStructure &nova_baking_structure = app_controller->getBakingStructure();
    if (use_gpu) {
#ifdef AXOMAE_USE_CUDA
      map_resources(render_scene_data);
      device::gpgpu::GPUContext context;
      device::gpgpu::get_current_context(context);
      auto callback = [](nova_baker_utils::render_scene_context &render_scene_data,
                         image::ImageHolder<float> &image_holder,
                         DisplayManager3D &disp_manager,
                         device::gpgpu::GPUContext &context) {
        disp_manager.haltRenderers();
        auto ret = do_progressive_render_gpu(render_scene_data, image_holder, std::move(context));
        disp_manager.resumeRenderers();
        return ret;
      };
      auto task = render_scene_data.thread_pool->addTask(
          threading::ALL_TASK, callback, std::ref(render_scene_data), std::ref(image_holder), std::ref(disp_manager), std::move(context));
      nova_baking_structure.on_cleanup = std::move(task);
#endif

    } else {
      auto callback =
          [](nova_baker_utils::render_scene_context &render_scene_data, image::ImageHolder<float> &image_holder, DisplayManager3D &disp_manager) {
            disp_manager.haltRenderers();
            auto ret = do_progressive_render(render_scene_data, image_holder);
            disp_manager.resumeRenderers();
            return ret;
          };
      auto task = render_scene_data.thread_pool->addTask(
          threading::ALL_TASK, callback, std::ref(render_scene_data), std::ref(image_holder), std::ref(disp_manager));
      nova_baking_structure.on_cleanup = std::move(task);
    }
  }

  static void ignore_skybox(const RendererInterface &renderer, bool ignore) {
    Scene &scene_ref = renderer.getScene();
    /* Cancel the transformation from the skybox , since it's a parent of the scene */
    scene_ref.ignoreSkyboxTransformation(ignore);
  }

  void Controller::do_nova_render(const ui_render_options &render_options, const engine_misc_options &misc_options) {

    GLViewer *realtime_viewer = display_manager.getRealtimeViewer();
    if (!(current_workspace->getContext() & UI_RENDERER_RASTER))
      return;
    if (!realtime_viewer)
      throw exception::InValidRendererException();

    const RendererInterface &renderer = realtime_viewer->getRenderer();
    const Camera *renderer_camera = renderer.getCamera();
    if (!renderer_camera) {
      LOG("The current renderer doesn't have a camera.", LogLevel::CRITICAL);
      ExceptionInfoBoxHandler::handle("No camera initialized in the renderer!", exception::CRITICAL);
      return;
    }

    nova::NovaResourceManager *resource_manager = display_manager.getNovaResourceManager();
    if (!resource_manager) {
      LOG("Resource manager is not initialized.", LogLevel::ERROR);
      ExceptionInfoBoxHandler::handle(exception::CRITICAL_TXT, exception::CRITICAL);
      return;
    }

    /* Meshes */
    ignore_skybox(renderer, true);
    const Scene &scene = renderer.getScene();
    const std::vector<Mesh *> &mesh_collection = scene.getMeshCollection();
    if (mesh_collection.empty()) {
      LOG("Scene is empty!", LogLevel::WARNING);
      ExceptionInfoBoxHandler::handle("Scene is empty!", exception::WARNING);
      ignore_skybox(renderer, false);
      return;
    }

    try {
      nova_baking_structure.reinitialize();
      nova_baker_utils::engine_data engine_data = generate_engine_data(mesh_collection, render_options, misc_options, *renderer_camera, renderer);
      std::unique_ptr<nova::NovaExceptionManager> exception_manager = std::make_unique<nova::NovaExceptionManager>();
      initialize_nova_manager(engine_data, *resource_manager);
      std::unique_ptr<NovaRenderEngineInterface> engine_instance = create_engine(engine_data);
      threading::ThreadPool *thread_pool = global_application_config->getThreadPool();
      setup_render_scene_data(render_options,
                              resource_manager,
                              std::move(exception_manager),
                              std::move(engine_instance),
                              getSharedCaches(),
                              nova_baking_structure,
                              thread_pool);
      start_baking(nova_baking_structure.render_context, this, nova_baking_structure.bake_buffers.image_holder, misc_options.use_gpu);
      ignore_skybox(renderer, false);
    } catch (const exception::CatastrophicFailureException &e) {
      ignore_skybox(renderer, false);
      LOG(e.what(), LogLevel::CRITICAL);
      cleanupNova();
      ExceptionInfoBoxHandler::handle(exception::CRITICAL_TXT, exception::CRITICAL);
    } catch (const std::exception &e) {
      ignore_skybox(renderer, false);
      LOG(e.what(), LogLevel::CRITICAL);
      cleanupNova();
      ExceptionInfoBoxHandler::handle(exception::CRITICAL_TXT, exception::CRITICAL);
    }
  }

  struct ui_inputs {
    int width, height, tiles_number, depth, samples_per_pixel;
    int render_type_mode_flag;
    bool use_gpu;
  };

  bool input_check(const ui_inputs &inputs) {
    if (inputs.width <= 0) {
      ExceptionInfoBoxHandler::handle("Invalid width.", exception::WARNING);
      return false;
    }
    if (inputs.height <= 0) {
      ExceptionInfoBoxHandler::handle("Invalid height.", exception::WARNING);
      return false;
    }
    if (inputs.tiles_number <= 0) {
      ExceptionInfoBoxHandler::handle("Invalid tile numbers.", exception::WARNING);
      return false;
    }
    if (inputs.depth < 1) {
      ExceptionInfoBoxHandler::handle("Invalid depth.", exception::WARNING);
      return false;
    }
    if (inputs.samples_per_pixel <= 0) {
      ExceptionInfoBoxHandler::handle("Invalid sample number.", exception::WARNING);
      return false;
    }
    if (inputs.use_gpu) {
#ifndef AXOMAE_USE_CUDA
      ExceptionInfoBoxHandler::handle("GPGPU not available. \nSee logs.(execute with --editor --verbose).", exception::WARNING);
      LOG("CUDA not found. Enable 'AXOMAE_USE_CUDA' in build if this platform has an Nvidia GPU.", LogLevel::ERROR);
      return false;
#endif
    }
    return true;
  }

  static int get_render_type_mode(const Ui::MainWindow &main_window_ui) {
    int render_type_flag = 0;

    /* color modes */
    if (main_window_ui.checkbox_rendermode_combined->isChecked())
      render_type_flag |= nova::integrator::COMBINED;
    if (main_window_ui.checkbox_rendermode_depth->isChecked())
      render_type_flag |= nova::integrator::DEPTH;
    if (main_window_ui.checkbox_rendermode_normal->isChecked())
      render_type_flag |= nova::integrator::NORMAL;
    if (main_window_ui.checkbox_rendermode_diffuse->isChecked())
      render_type_flag |= nova::integrator::DIFFUSE;
    if (main_window_ui.checkbox_rendermode_emissive->isChecked())
      render_type_flag |= nova::integrator::EMISSIVE;
    if (main_window_ui.checkbox_rendermode_specular->isChecked())
      render_type_flag |= nova::integrator::SPECULAR;

    /* renderer type */
    std::string render_type = main_window_ui.combobox_enginetype_selector->currentText().toStdString();
    if (render_type == "Path")
      render_type_flag |= nova::integrator::PATH;
    else if (render_type == "Bidirectional Path")
      render_type_flag |= nova::integrator::BIPATH;
    else if (render_type == "Spectral")
      render_type_flag |= nova::integrator::SPECTRAL;
    else if (render_type == "Metropolis")
      render_type_flag |= nova::integrator::METROPOLIS;
    else if (render_type == "Photon Mapping")
      render_type_flag |= nova::integrator::PHOTON;
    else if (render_type == "Marching")
      render_type_flag |= nova::integrator::MARCHING;
    else if (render_type == "Hybrid")
      render_type_flag |= nova::integrator::HYBRID;
    else if (render_type == "Voxel")
      render_type_flag |= nova::integrator::VOXEL;
    return render_type_flag;
  }

  void Controller::slot_nova_start_bake() {
    ui_inputs inputs{};
    inputs.width = main_window_ui.nova_bake_width->value();
    inputs.height = main_window_ui.nova_bake_height->value();
    inputs.samples_per_pixel = main_window_ui.spinbox_sample_per_pixel->value();
    inputs.depth = main_window_ui.spinbox_render_depth->value();
    inputs.tiles_number = main_window_ui.spinbox_tile_number->value();
    inputs.render_type_mode_flag = get_render_type_mode(main_window_ui);
    bool flip_v = main_window_ui.checkbox_flip_y_axis->isChecked();
    inputs.use_gpu = main_window_ui.use_gpu->isChecked();
    if (!input_check(inputs))
      return;

    ui_render_options render_options{};
    engine_misc_options misc_options{};
    render_options.aa_samples = 1;
    render_options.max_depth = inputs.depth;
    render_options.max_samples = inputs.samples_per_pixel;
    render_options.tiles = inputs.tiles_number;
    render_options.width = inputs.width;
    render_options.height = inputs.height;

    misc_options.use_gpu = inputs.use_gpu;
    misc_options.engine_type_flag = inputs.render_type_mode_flag;
    misc_options.flip_v = flip_v;
    do_nova_render(render_options, misc_options);
  }

  void Controller::novaStopBake() {
    /* Stop the threads. */
    auto &nova_resource_manager = nova_baking_structure.render_context.nova_resource_manager;
    if (nova_resource_manager) {
      nova_resource_manager->getEngineData().is_rendering = false;
      /* Cleanup gpu resources */
      nova_resource_manager->getShapeData().releaseResources();
      nova_resource_manager->getTexturesData().releaseResources();
    }
    if (global_application_config && global_application_config->getThreadPool()) {
      /* Empty scheduler list. */
      global_application_config->getThreadPool()->emptyQueue(NOVABAKE_POOL_TAG);
      /* Synchronize the threads. */
      global_application_config->getThreadPool()->fence(NOVABAKE_POOL_TAG);
    }
    display_manager.resumeRenderers();
  }

  void Controller::cleanupNova() {
    novaStopBake();
    nova_baking_structure.reinitialize();
  }

  void Controller::slot_nova_stop_bake() { novaStopBake(); }

  void Controller::slot_nova_save_bake(const image::ImageHolder<float> &image_holder) {
    std::string filename = spawnSaveFileDialogueWidget();
    if (!filename.empty()) {
      IO::Loader loader(getProgress());
      loader.writeHdr(filename.c_str(), image_holder, false);
    }
  }

}  // namespace controller
