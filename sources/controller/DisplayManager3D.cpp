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

  void DisplayManager3D::setNewScene(SceneChangeData scene_data_mv) {
    SceneChangeData scene_data = std::move(scene_data_mv);
    prepareSceneChange();
    scene_data.nova_resource_manager = nova_resource_manager.get();

    realtime_viewer->setNewScene(scene_data);
    nova_viewer->setNewScene(scene_data);
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