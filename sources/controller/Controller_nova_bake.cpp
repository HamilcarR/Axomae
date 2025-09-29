#include "Config.h"
#include "EnvmapTextureManager.h"
#include "ExceptionHandlerUI.h"
#include "GUIWindow.h"
#include "Image.h"
#include "TextureViewerWidget.h"
#include "WorkspaceTracker.h"
#include "api_common.h"
#include "api_datastructures.h"
#include "engine/nova_exception.h"
#include "integrator/Integrator.h"
#include "manager/ManagerInternalStructs.h"
#include "manager/NovaResourceManager.h"
#include "nova/bake.h"
#include "nova/bake_render_data.h"
#include <QFileDialog>
#include <future>
#include <internal/common/exception/GenericException.h>
#include <internal/debug/Logger.h>
#include <internal/debug/debug_utils.h>
#include <internal/macro/exception_macros.h>
#include <internal/macro/project_macros.h>
#include <qwidget.h>
#include <string>
#include <thread>

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
                           const ui_render_options &interface_eng_options,
                           const engine_misc_options &misc_options) {
    engine_data.aa_samples = interface_eng_options.aa_samples;
    engine_data.depth_max = interface_eng_options.max_depth;
    engine_data.samples_max = interface_eng_options.max_samples;
    engine_data.num_tiles_w = engine_data.num_tiles_h = interface_eng_options.tiles;
    engine_data.samples_increment = 0;
    engine_data.engine_type_flag = misc_options.engine_type_flag;
    engine_data.flip_v = misc_options.flip_v;
    engine_data.threadpool_tag = NOVABAKE_POOL_TAG;
    engine_data.use_gpu = misc_options.use_gpu;
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
    setup_envmaps(renderer.getCurrentEnvmapId(), engine_data.environment_maps);
    return engine_data;
  }

  static void allocate_display_widget_buffer(image::ThumbnailImageHolder<float> &image_holder, const nova::RenderBuffer &rb) {

    unsigned COLOR_CHANS = rb.getChannel(nova::texture::COLOR);
    size_t color_buffer_size = rb.getWidth() * rb.getHeight() * COLOR_CHANS;
    image::Metadata metadata;
    metadata.channels = COLOR_CHANS;
    metadata.color_corrected = false;
    metadata.format = "hdr";
    metadata.height = rb.getHeight();
    metadata.width = rb.getWidth();
    metadata.is_hdr = true;
    const nova::Framebuffer &framebuffer = rb.getFramebuffer();
    image_holder = image::ThumbnailImageHolder(framebuffer.color_buffer, metadata, nullptr, false);
  }

  static nova::HdrBufferStruct setup_render_buffer(const ui_render_options &render_options, nova::RenderBuffer &render_buffer) {
    render_buffer.resize(render_options.width, render_options.height);
    return render_buffer.getRenderBuffers();
  }

  /**************************************************************************************************************/

  static std::unique_ptr<HdrNovaRenderViewerWidget> texture_display_render(Controller *app_controller, nova::RenderBuffer *render_buffer) {
    return std::make_unique<HdrNovaRenderViewerWidget>(render_buffer, app_controller, true, app_controller);
  }

  /* returns true if exception needs to shutdown renderer  */
  static bool on_exception_shutdown(const nova::NovaExceptionManager &exception_manager) {
    using namespace nova::exception;
    if (nova::get_error_status(exception_manager) != 0) {
      auto exception_list = nova::get_error_list(exception_manager);
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

  static void display_render(controller::Controller *app_controller, nova::RenderBuffer *render_buffer) {
    /* Creates a texture widget that will display the rendered image */
    nova_baker_utils::NovaBakingStructure &nova_baking_structure = app_controller->getBakingStructure();
    auto &texture_widget = nova_baking_structure.spawned_window;
    texture_widget = texture_display_render(app_controller, render_buffer);
    if (!texture_widget)
      throw exception::NullTextureWidgetException();
    QWidget *img_ptr = texture_widget.get();
    img_ptr->show();
  }

  template<class T>
  class CudaContext : public nova::EngineUserCallbackHandler {
    T gpu_context;

   public:
    CudaContext(T &&ctx) : gpu_context(std::move(ctx)) {}
    void framePreRenderRoutine() override { device::gpgpu::apply_context(gpu_context); }
    void framePostRenderRoutine() override { device::gpgpu::register_context(gpu_context); }
  };

  static nova::EngineUserCallbackHandlerPtr register_nova_callbacks(const nova::Engine &engine, QWidget *display_window) {
    const nova::RenderOptions *render_options = engine.getRenderOptions();

#ifdef AXOMAE_USE_CUDA
    if (render_options->isUsingGpu()) {
      device::gpgpu::GPUContext context;
      device::gpgpu::register_context(context);
      return std::make_unique<CudaContext<device::gpgpu::GPUContext>>(std::move(context));
    }
#endif
    return nullptr;
  }

  static void start_baking(nova::Engine &nova_engine, Controller *app_controller, nova_baker_utils::NovaBakingStructure &data) {
    nova::RenderBuffer *nova_buffers = nova_engine.getRenderBuffers();
    display_render(app_controller, nova_buffers);
    nova::EngineUserCallbackHandlerPtr callback = register_nova_callbacks(nova_engine, data.spawned_window.get());
    nova_engine.setUserCallback(std::move(callback));

    nova_engine.prepareRender();
    nova_engine.startRender();
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

    nova::Engine *nova_engine = display_manager.getNovaEngineInstance();
    if (!nova_engine) {
      LOG("Nova engine is not initialized.", LogLevel::ERROR);
      ExceptionInfoBoxHandler::handle(exception::CRITICAL_TXT, exception::CRITICAL);
      return;
    }

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

      initialize_engine(engine_data, *nova_engine);
      nova::RenderBuffer *render_buffer = nova_engine->getRenderBuffers();
      nova::HdrBufferStruct target_buffers = setup_render_buffer(render_options, *render_buffer);

      allocate_display_widget_buffer(nova_baking_structure.bake_buffers.image_holder, *render_buffer);
      start_baking(*nova_engine, this, nova_baking_structure);
    } catch (const exception::CatastrophicFailureException &e) {
      LOG(e.what(), LogLevel::CRITICAL);
      cleanupNova();
      ExceptionInfoBoxHandler::handle(exception::CRITICAL_TXT, exception::CRITICAL);
    } catch (const std::exception &e) {
      LOG(e.what(), LogLevel::CRITICAL);
      cleanupNova();
      ExceptionInfoBoxHandler::handle(exception::CRITICAL_TXT, exception::CRITICAL);
    }
    ignore_skybox(renderer, false);
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

    nova::Engine *engine = display_manager.getNovaEngineInstance();
    engine->stopRender();
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
