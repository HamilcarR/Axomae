
#include "Config.h"
#include "GUIWindow.h"
#include "ImageImporter.h"
#include "Logger.h"
#include "WorkspaceTracker.h"
#include "manager/NovaResourceManager.h"
#include "nova/bake.h"
#include <QFileDialog>

static constexpr int AA_SAMPLES = 8;
static constexpr int MAX_SAMPLES = 1000;
static constexpr int MAX_DEPTH = 5;
static constexpr int TILES = 22;
static std::mutex mutex;
/**************************************************************************************************************/
namespace controller {
  static void cleanup() {}
  static void init_camera(nova_baker_utils::engine_data &engine_data, const Camera &renderer_camera, float render_width, float render_height) {
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

  static void init_scene_transfo(nova_baker_utils::engine_data &engine_data, const Camera &renderer_camera) {
    /* Scene transformations */
    nova_baker_utils::scene_transform_data scene_transfo{};
    scene_transfo.root_rotation = renderer_camera.getSceneRotationMatrix();
    scene_transfo.root_transformation = renderer_camera.getLocalModelMatrix();
    scene_transfo.root_translation = renderer_camera.getSceneTranslationMatrix();

    engine_data.scene = scene_transfo;
  }

  static void init_envmap(nova_baker_utils::engine_data &engine_data, const RendererInterface &renderer) {
    /* Environment map */
    nova_baker_utils::scene_envmap envmap{};
    image::ImageHolder<float> *env = renderer.getCurrentEnvmapId().currentMutableEnvmapMetadata();
    envmap.hdr_envmap = env;

    engine_data.envmap = envmap;
  }

  struct ui_render_options {
    int aa_samples;
    int max_samples;
    int max_depth;
    int tiles;
  };
  struct engine_misc_options {
    bool *cancel_ptr;
    int engine_type_flag;
    bool flip_v;
  };
  static void init_engine(nova_baker_utils::engine_data &engine_data,
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
  }
  /**************************************************************************************************************/

  void Controller::save_bake(const image::ImageHolder<float> &image_holder) {
    std::string filename = spawnSaveFileDialogueWidget();
    if (!filename.empty()) {
      IO::Loader loader(getProgress());
      loader.writeHdr(filename.c_str(), image_holder, false);
    }
  }

  void save_bake(const image::ImageHolder<float> &image_holder, const std::string &filename) {
    if (!filename.empty()) {
      IO::Loader loader(nullptr);
      loader.writeHdr(filename.c_str(), image_holder, false);
    }
  }

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
    // image_holder.flip_v();
  }

  static void do_progressive_render(nova_baker_utils::render_scene_data &render_scene_data, image::ImageHolder<float> &image_holder) {
    int i = 1;
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
      nova_baker_utils::synchronize_render_threads(render_scene_data);
      color_correct_buffers(render_scene_data.buffers.get(), image_holder, (float)i);
      i++;
    }
    std::lock_guard lock(mutex);
    save_bake(image_holder, "4k_bananum");
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

    /* Creates a texture widget that will display the rendered image */
    auto &texture_widget = app_controller->getBakingStructure().spawned_window;
    texture_widget = texture_display_render(app_controller, image_holder);
    QWidget *img_ptr = texture_widget.get();
    img_ptr->show();
  }

  void Controller::nova_baking() {
    nova_baking_structure.reinitialize();
    if (!(current_workspace->getContext() & UI_RENDERER_RASTER))
      return;
    if (!realtime_viewer)
      return;
    int width = main_window_ui.nova_bake_width->value();
    int height = main_window_ui.nova_bake_height->value();

    const RendererInterface &renderer = realtime_viewer->getRenderer();
    const Camera *renderer_camera = renderer.getCamera();
    if (!renderer_camera) {
      LOG("The current renderer doesn't have a camera.", LogLevel::ERROR);
      return;
    }

    std::unique_ptr<nova::NovaResourceManager> manager = std::make_unique<nova::NovaResourceManager>();
    nova_baker_utils::engine_data engine_data{};
    /* Meshes */
    Scene &scene_ref = renderer.getScene();
    auto &skybox = dynamic_cast<SceneNodeInterface &>(scene_ref.getSkybox());
    skybox.ignoreTransformation(true);
    scene_ref.updateTree();
    const Scene &scene = renderer.getScene();
    const std::vector<Mesh *> &mesh_collection = scene.getMeshCollection();
    if (mesh_collection.empty()) {
      LOG("Scene not initialized.", LogLevel::INFO);
      return;
    }

    engine_data.mesh_list = &mesh_collection;

    ui_render_options render_options;
    engine_misc_options misc_options;
    render_options.aa_samples = AA_SAMPLES;
    render_options.max_depth = MAX_DEPTH;
    render_options.max_samples = MAX_SAMPLES;
    render_options.tiles = TILES;

    bool &b = nova_baking_structure.stop;
    misc_options.cancel_ptr = &b;
    misc_options.engine_type_flag = nova_baker_utils::ENGINE_TYPE::RGB;
    misc_options.flip_v = true;

    init_engine(engine_data, render_options, misc_options);
    init_camera(engine_data, *renderer_camera, width, height);
    init_scene_transfo(engine_data, *renderer_camera);
    init_envmap(engine_data, renderer);
    nova_baker_utils::initialize_manager(engine_data, *manager);

    /* buffers*/
    image::ThumbnailImageHolder<float> &image_holder = nova_baking_structure.bake_buffers.image_holder;
    std::vector<float> &partial = nova_baking_structure.bake_buffers.partial;
    std::vector<float> &accumulator = nova_baking_structure.bake_buffers.accumulator;
    image_holder.data.resize(width * height * 4);
    partial.resize(width * height * 4);
    accumulator.resize(width * height * 4);

    image_holder.metadata.channels = 4;
    image_holder.metadata.color_corrected = false;
    image_holder.metadata.format = "hdr";
    image_holder.metadata.height = height;
    image_holder.metadata.width = width;
    image_holder.metadata.is_hdr = true;
    std::unique_ptr<nova::HdrBufferStruct> buffers = std::make_unique<nova::HdrBufferStruct>();

    buffers->partial_buffer = partial.data();
    buffers->accumulator_buffer = accumulator.data();

    /* Engine type */
    std::unique_ptr<NovaRenderEngineInterface> engine_instance = create_engine(engine_data);
    nova_baking_structure.nova_render_scene = nova_baker_utils::render_scene_data{};
    nova_baker_utils::render_scene_data &render_scene_data = nova_baking_structure.nova_render_scene;
    render_scene_data.buffers = std::move(buffers);
    render_scene_data.width = width;
    render_scene_data.height = height;
    render_scene_data.engine_instance = std::move(engine_instance);
    render_scene_data.nova_resource_manager = std::move(manager);
    render_scene_data.thread_pool = global_application_config->getThreadPool();

    start_baking(render_scene_data, this, image_holder);

    skybox.ignoreTransformation(false);
    scene_ref.updateTree();
    // save_bake(image_holder);
  }
}  // namespace controller
