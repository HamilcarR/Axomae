
#include "Config.h"
#include "ExceptionHandlerUI.h"
#include "GUIWindow.h"
#include "GenericException.h"
#include "Logger.h"
#include "WorkspaceTracker.h"
#include "exception_macros.h"
#include "manager/NovaResourceManager.h"
#include "nova/bake.h"
#include <QFileDialog>

namespace exception {
  GENERIC_EXCEPT_DEFINE(InValidCameraException, "No valid camera present in the scene , or the camera hasn't been initialized.", WARNING)
  GENERIC_EXCEPT_DEFINE(InValidRendererException, "The rasterizer hasn't been intialized.", WARNING)
  GENERIC_EXCEPT_DEFINE(InvalidSceneException, "Current scene is not initialized.", WARNING)
  GENERIC_EXCEPT_DEFINE(InvalidInputException, "Invalid input.", INFO)
  GENERIC_EXCEPT_DEFINE(InvalidRenderBufferStateException, "Render buffers are in an invalid state.", CRITICAL)
  GENERIC_EXCEPT_DEFINE(InvalidSceneNodeTypeException, "Scene node type discrepancy.", CRITICAL)
  GENERIC_EXCEPT_DEFINE(NullTextureWidgetException, "Failed to create output texture.", CRITICAL)
  const char *const CRITICAL_TXT = "A critical error occured. \nSee the logs for more precision.";

  class ErrHandler {
   public:
    static void handle(const std::string &message, SEVERITY severity) { controller::ExceptionHandlerUI::launchInfoBox(message, severity); }
    static void handle(const ExceptionData &e) {
      SEVERITY s = controller::ExceptionHandlerUI::intToSeverity(e.getSeverity());
      controller::ExceptionHandlerUI::launchInfoBox(e.getErrorMessage(), s);
    }
  };
}  // namespace exception

static std::mutex mutex;
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
    bool *cancel_ptr;
    int engine_type_flag;
    bool flip_v;
  };

  static void setup_camera(nova_baker_utils::engine_data &engine_data, const Camera &renderer_camera, float render_width, float render_height) {
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

    engine_data.camera = camera_data;
  }

  static void setup_scene_transfo(nova_baker_utils::engine_data &engine_data, const Camera &renderer_camera) {
    /* Scene transformations */
    nova_baker_utils::scene_transform_data scene_transfo{};
    scene_transfo.root_rotation = renderer_camera.getSceneRotationMatrix();
    scene_transfo.root_transformation = renderer_camera.getLocalModelMatrix();
    scene_transfo.root_translation = renderer_camera.getSceneTranslationMatrix();

    engine_data.scene = scene_transfo;
  }

  static void setup_environment_map(nova_baker_utils::engine_data &engine_data, const RendererInterface &renderer) {
    /* Environment map */
    nova_baker_utils::scene_envmap envmap{};
    image::ImageHolder<float> *env = renderer.getCurrentEnvmapId().currentMutableEnvmapMetadata();
    envmap.hdr_envmap = env;

    engine_data.envmap = envmap;
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
    engine_data.stop_render_ptr = misc_options.cancel_ptr;
    engine_data.flip_v = misc_options.flip_v;
    engine_data.threadpool_tag = NOVABAKE_POOL_TAG;
  }

  std::unique_ptr<nova::HdrBufferStruct> generate_buffers(NovaBakingStructure &nova_baking_structure, const ui_render_options &render_options) {

    /* buffers*/
    image::ThumbnailImageHolder<float> &image_holder = nova_baking_structure.bake_buffers.image_holder;
    std::vector<float> &partial = nova_baking_structure.bake_buffers.partial;
    std::vector<float> &accumulator = nova_baking_structure.bake_buffers.accumulator;
    image_holder.data.resize(render_options.width * render_options.height * 4);
    partial.resize(render_options.width * render_options.height * 4);
    accumulator.resize(render_options.width * render_options.height * 4);

    image_holder.metadata.channels = 4;
    image_holder.metadata.color_corrected = false;
    image_holder.metadata.format = "hdr";
    image_holder.metadata.height = render_options.height;
    image_holder.metadata.width = render_options.width;
    image_holder.metadata.is_hdr = true;
    std::unique_ptr<nova::HdrBufferStruct> buffers = std::make_unique<nova::HdrBufferStruct>();

    buffers->partial_buffer = partial.data();
    buffers->accumulator_buffer = accumulator.data();
    return buffers;
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
    setup_scene_transfo(engine_data, renderer_camera);
    setup_environment_map(engine_data, renderer);
    return engine_data;
  }

  void setup_render_scene_data(const ui_render_options &render_options,
                               std::unique_ptr<nova::NovaResourceManager> &&manager,
                               std::unique_ptr<NovaRenderEngineInterface> &&engine_instance,
                               NovaBakingStructure &nova_baking_structure,
                               threading::ThreadPool *thread_pool) {
    /* Engine type */
    nova_baking_structure.nova_render_scene = nova_baker_utils::render_scene_data{};
    nova_baker_utils::render_scene_data &render_scene_data = nova_baking_structure.nova_render_scene;
    render_scene_data.buffers = generate_buffers(nova_baking_structure, render_options);
    render_scene_data.width = render_options.width;
    render_scene_data.height = render_options.height;
    render_scene_data.engine_instance = std::move(engine_instance);
    render_scene_data.nova_resource_manager = std::move(manager);
    render_scene_data.thread_pool = thread_pool;
  }

  /**************************************************************************************************************/

  static std::unique_ptr<HdrRenderViewerWidget> texture_display_render(Controller *app_controller, const image::ThumbnailImageHolder<float> &img) {
    return std::make_unique<HdrRenderViewerWidget>(&img, app_controller);
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

  static void do_progressive_render(nova_baker_utils::render_scene_data &render_scene_data, image::ImageHolder<float> &image_holder) {
    int i = 1;
    int MAX_DEPTH = render_scene_data.nova_resource_manager->getEngineData().getMaxDepth();
    const int N = render_scene_data.nova_resource_manager->getEngineData().getMaxSamples();
    const int smax = math::calculus::compute_serie_term(N);
    const bool *stop_ptr = render_scene_data.nova_resource_manager->getEngineData().getCancelPtr();
    while (i < smax && !(*stop_ptr)) {
      render_scene_data.nova_resource_manager->getEngineData().setSampleIncrement(i);
      render_scene_data.nova_resource_manager->getEngineData().setMaxDepth(
          render_scene_data.nova_resource_manager->getEngineData().getMaxDepth() < MAX_DEPTH ?
              render_scene_data.nova_resource_manager->getEngineData().getMaxDepth() + 1 :
              MAX_DEPTH);

      nova_baker_utils::bake_scene(render_scene_data);
      nova_baker_utils::synchronize_render_threads(render_scene_data, NOVABAKE_POOL_TAG);
      color_correct_buffers(render_scene_data.buffers.get(), image_holder, (float)i);
      i++;
    }
  }

  static void display_render(controller::Controller *app_controller, image::ThumbnailImageHolder<float> &image_holder) {
    /* Creates a texture widget that will display the rendered image */
    NovaBakingStructure &nova_baking_structure = app_controller->getBakingStructure();
    auto &texture_widget = nova_baking_structure.spawned_window;
    texture_widget = texture_display_render(app_controller, image_holder);
    if (!texture_widget)
      throw exception::NullTextureWidgetException();
    QWidget *img_ptr = texture_widget.get();
    img_ptr->show();
  }

  static void start_baking(nova_baker_utils::render_scene_data &render_scene_data,
                           Controller *app_controller,
                           image::ThumbnailImageHolder<float> &image_holder) {

    auto callback = [](nova_baker_utils::render_scene_data &render_scene_data, image::ImageHolder<float> &image_holder) {
      do_progressive_render(render_scene_data, image_holder);
    };

    /* Creates a rendering thread . Will be moved to the baking structure ,
     * which will take care of managing it's lifetime */
    std::thread th(callback, std::ref(render_scene_data), std::ref(image_holder));
    NovaBakingStructure &nova_baking_structure = app_controller->getBakingStructure();
    nova_baking_structure.rendering_thread = std::move(th);
    try {
      display_render(app_controller, image_holder);
    } catch (const std::exception &e) {
      throw;
    }
  }

  static void ignore_skybox(const RendererInterface &renderer, bool ignore) {
    Scene &scene_ref = renderer.getScene();
    /* Cancel the transformation from the skybox , since it's a parent of the scene */
    scene_ref.ignoreSkyboxTransformation(ignore);
  }

  void Controller::do_nova_render(const ui_render_options &render_options, const engine_misc_options &misc_options) {

    if (!(current_workspace->getContext() & UI_RENDERER_RASTER))
      return;
    if (!realtime_viewer)
      throw exception::InValidRendererException();

    const RendererInterface &renderer = realtime_viewer->getRenderer();
    const Camera *renderer_camera = renderer.getCamera();
    if (!renderer_camera) {
      LOG("The current renderer doesn't have a camera.", LogLevel::CRITICAL);
      exception::ErrHandler::handle("No camera initialized in the renderer!", exception::CRITICAL);
      return;
    }

    /* Meshes */
    ignore_skybox(renderer, true);
    const Scene &scene = renderer.getScene();
    const std::vector<Mesh *> &mesh_collection = scene.getMeshCollection();
    if (mesh_collection.empty()) {
      LOG("Scene is empty!", LogLevel::WARNING);
      exception::ErrHandler::handle("Scene is empty!", exception::WARNING);
      ignore_skybox(renderer, false);
      return;
    }
    try {
      nova_baking_structure.reinitialize();
      nova_baker_utils::engine_data engine_data = generate_engine_data(mesh_collection, render_options, misc_options, *renderer_camera, renderer);
      std::unique_ptr<nova::NovaResourceManager> manager = std::make_unique<nova::NovaResourceManager>();
      nova_baker_utils::initialize_manager(engine_data, *manager);
      std::unique_ptr<NovaRenderEngineInterface> engine_instance = create_engine(engine_data);
      threading::ThreadPool *thread_pool = global_application_config->getThreadPool();
      setup_render_scene_data(render_options, std::move(manager), std::move(engine_instance), nova_baking_structure, thread_pool);
      start_baking(nova_baking_structure.nova_render_scene, this, nova_baking_structure.bake_buffers.image_holder);
      ignore_skybox(renderer, false);
    } catch (const exception::CatastrophicFailureException &e) {
      ignore_skybox(renderer, false);
      LOG(e.what(), LogLevel::CRITICAL);
      cleanupNova();
      exception::ErrHandler::handle(exception::CRITICAL_TXT, exception::CRITICAL);
    } catch (const std::exception &e) {
      ignore_skybox(renderer, false);
      LOG(e.what(), LogLevel::CRITICAL);
      cleanupNova();
      exception::ErrHandler::handle(exception::CRITICAL_TXT, exception::CRITICAL);
    }
  }

  struct ui_inputs {
    int width, height, tiles_number, depth, samples_per_pixel;
  };

  bool input_check(const ui_inputs &inputs) {
    if (inputs.width <= 0) {
      exception::ErrHandler::handle("Invalid width.", exception::WARNING);
      return false;
    }
    if (inputs.height <= 0) {
      exception::ErrHandler::handle("Invalid height.", exception::WARNING);
      return false;
    }
    if (inputs.tiles_number <= 0) {
      exception::ErrHandler::handle("Invalid tile numbers.", exception::WARNING);
      return false;
    }
    if (inputs.depth < 1) {
      exception::ErrHandler::handle("Invalid depth.", exception::WARNING);
      return false;
    }
    if (inputs.samples_per_pixel <= 0) {
      exception::ErrHandler::handle("Invalid sample number.", exception::WARNING);
      return false;
    }
    return true;
  }

  void Controller::slot_nova_start_bake() {
    ui_inputs inputs{};
    inputs.width = main_window_ui.nova_bake_width->value();
    inputs.height = main_window_ui.nova_bake_height->value();
    inputs.samples_per_pixel = main_window_ui.spinbox_sample_per_pixel->value();
    inputs.depth = main_window_ui.spinbox_render_depth->value();
    inputs.tiles_number = main_window_ui.spinbox_tile_number->value();
    bool flip_v = main_window_ui.checkbox_flip_y_axis->isChecked();
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

    nova_baking_structure.stop = false;
    bool &b = nova_baking_structure.stop;
    misc_options.cancel_ptr = &b;
    misc_options.engine_type_flag = nova_baker_utils::ENGINE_TYPE::RGB;
    misc_options.flip_v = flip_v;
    do_nova_render(render_options, misc_options);
  }

  void Controller::novaStopBake() {
    /* Stop the threads. */
    nova_baking_structure.stop = true;
    if (global_application_config && global_application_config->getThreadPool()) {
      /* Empty scheduler list. */
      global_application_config->getThreadPool()->emptyQueue(NOVABAKE_POOL_TAG);
      /* Synchronize the threads. */
      global_application_config->getThreadPool()->fence(NOVABAKE_POOL_TAG);
    }
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
