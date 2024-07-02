#include "Config.h"
#include "GUIWindow.h"
#include "ImageImporter.h"
#include "Logger.h"
#include "WorkspaceTracker.h"
#include "manager/NovaResourceManager.h"
#include "nova/bake.h"

#include <QFileDialog>
namespace controller {
  /**************************************************************************************************************/
  void Controller::nova_baking() {
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
    nova::NovaResourceManager manager;
    nova_baker_utils::engine_data engine_data;
    /* Meshes */
    const Scene &scene = renderer.getScene();
    std::vector<Mesh *> mesh_collection = scene.getMeshCollectionPtr();
    if (mesh_collection.empty()) {
      LOG("Scene not initialized.", LogLevel::INFO);
      return;
    }
    engine_data.mesh_list = &mesh_collection;

    /* Camera */
    nova_baker_utils::camera_data camera_data{};
    camera_data.position = renderer_camera->getPosition();
    camera_data.direction = renderer_camera->getDirection();
    camera_data.up_vector = renderer_camera->getUpVector();
    camera_data.projection = renderer_camera->getProjection();
    camera_data.view = renderer_camera->getTransformedView();

    engine_data.camera = camera_data;

    /* Scene transformations */
    nova_baker_utils::scene_transform_data scene_transfo{};
    scene_transfo.root_rotation = renderer_camera->getSceneRotationMatrix();
    scene_transfo.root_transformation = renderer_camera->getLocalModelMatrix();
    scene_transfo.root_translation = renderer_camera->getSceneTranslationMatrix();

    engine_data.scene = scene_transfo;

    /* Environment map */
    nova_baker_utils::scene_envmap envmap{};
    image::ImageHolder<float> *env = renderer.getCurrentEnvmapId().currentMutableEnvmapMetadata();
    envmap.hdr_envmap = env;

    engine_data.envmap = envmap;

    /* engine data */
    int id = stop_list.size();
    stop_list.insert(std::pair(id, false));
    bool &b = stop_list.at(id);
    engine_data.aa_samples = 8;
    engine_data.depth_max = 2;
    engine_data.samples_max = engine_data.samples_increment = 10;
    engine_data.num_tiles_h = 20;
    engine_data.num_tiles_w = 20;
    engine_data.engine_type_flag = nova_baker_utils::ENGINE_TYPE::RGB;
    engine_data.stop_render_ptr = &b;
    nova_baker_utils::initialize_manager(engine_data, manager);

    /* buffers*/
    image::ImageHolder<float> image_holder;
    image_holder.data.resize(width * height * 4);
    image_holder.metadata.channels = 4;
    image_holder.metadata.color_corrected = false;
    image_holder.metadata.format = "hdr";
    image_holder.metadata.height = height;
    image_holder.metadata.width = width;
    image_holder.metadata.is_hdr = true;
    nova::HdrBufferStruct buffers;
    buffers.partial_buffer = image_holder.data.data();
    std::vector<float> accum = image_holder.data;
    buffers.accumulator_buffer = accum.data();

    /* Engine type */
    std::unique_ptr<NovaRenderEngineInterface> engine_instance = create_engine(engine_data);
    nova_baker_utils::render_scene_data render_scene_data;
    render_scene_data.buffers = &buffers;
    render_scene_data.width = width;
    render_scene_data.height = height;
    render_scene_data.engine_instance = engine_instance.get();
    render_scene_data.nova_resource_manager = &manager;
    render_scene_data.thread_pool = global_application_config->getThreadPool();
    nova_baker_utils::bake_scene(render_scene_data);
    nova_baker_utils::synchronize_render_threads(render_scene_data);

    QString filename = QFileDialog::getSaveFileName(this, tr("Save files"), "./", tr("All Files (*)"));

    IO::Loader loader(progress_manager.get());
    loader.writeHdr(filename.toStdString().c_str(), image_holder, true);
  }
}  // namespace controller