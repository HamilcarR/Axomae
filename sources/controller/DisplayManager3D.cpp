#include "DisplayManager3D.h"

#include "Config.h"
#include "RenderPipeline.h"
#include "nova/bake.h"

namespace exception {
  class SceneTreeInitializationException : public CatastrophicFailureException {
   public:
    static constexpr const char *text = "Realtime viewer not initialized, cannot retrieve scene tree.";
    SceneTreeInitializationException() { saveErrorString(text); }
  };
}  // namespace exception

namespace controller {
  void DisplayManager3D::init(Ui::MainWindow &main_window_ui,
                              ApplicationConfig *global_application_config,
                              OperatorProgressStatus *progress_manager) {

    /* Realtime renderer initialization*/
    realtime_viewer = main_window_ui.renderer_view;

    realtime_viewer->getRenderer().getRenderPipeline().setProgressManager(progress_manager);
    realtime_viewer->setProgressManager(progress_manager);
    realtime_viewer->setApplicationConfig(global_application_config);

    /* Nova raytracer */
    nova_viewer = main_window_ui.nova_viewer->getViewer();
    nova_viewer->renderOnTimer(0);
    nova_viewer->getRenderer().getRenderPipeline().setProgressManager(progress_manager);
    nova_viewer->setProgressManager(progress_manager);
    nova_viewer->setApplicationConfig(global_application_config);

    nova_resource_manager = std::make_unique<nova::NovaResourceManager>();
  }

  static void add_caches_addresses(nova::device_shared_caches_t &shared_caches,
                                   const nova_baker_utils::bake_buffers_storage_t &bake_buffers_storage) {
    /* Only texture buffers are taken into account for now.*/
    shared_caches.addSharedCacheAddress(bake_buffers_storage.texture_buffers.image_alloc_buffer);
  }

  static void notify_simple_progress(const char *str, IProgressManager *progress_manager) {
    if (progress_manager) {
      progress_manager->initProgress(str, 100.f);
      progress_manager->setCurrent(90);
      progress_manager->notifyProgress();
    }
  }

  void DisplayManager3D::setNewScene(SceneChangeData scene_data_mv, ProgressStatus *progress_status) {
    SceneChangeData scene_data = std::move(scene_data_mv);
    prepareSceneChange();
    scene_data.nova_resource_manager = nova_resource_manager.get();
    IProgressManager progress_manager;
    progress_manager.setProgressManager(progress_status);
    nova_resource_manager->clearResources();
    notify_simple_progress("Initializing Nova , building BVH and allocating shared GPU caches", &progress_manager);
    bake_buffers_storage = nova_baker_utils::build_scene(scene_data.mesh_list, *nova_resource_manager);
    add_caches_addresses(shared_caches, bake_buffers_storage);
    /* Build acceleration. */
    nova::aggregate::Accelerator accelerator = nova_baker_utils::build_performance_acceleration_structure(
        nova_resource_manager->getPrimitiveData().get_primitives());
    nova_resource_manager->setAccelerationStructure(accelerator);

    realtime_viewer->setNewScene(scene_data);
    nova_viewer->setNewScene(scene_data);
    progress_manager.reset();
  }

  void DisplayManager3D::prepareSceneChange() {
    realtime_viewer->prepareRendererSceneChange();
    nova_viewer->prepareRendererSceneChange();
  }

  SceneTree &DisplayManager3D::getSceneTree() const {
    if (!realtime_viewer) {
      throw exception::SceneTreeInitializationException();
    }
    return realtime_viewer->getRenderer().getScene().getSceneTreeRef();
  }

}  // namespace controller