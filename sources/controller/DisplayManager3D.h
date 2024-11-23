#ifndef DISPLAYMANAGER3D_H
#define DISPLAYMANAGER3D_H
#include "aggregate/nova_acceleration.h"
#include "ui_main_window.h"

#include <manager/NovaResourceManager.h>

class ApplicationConfig;

namespace controller {
  class OperatorProgressStatus;

  class DisplayManager3D {

   private:
    /* The 3D model viewer */
    GLViewer *realtime_viewer;
    /* Raytracing engine display*/
    GLViewer *nova_viewer;
    std::unique_ptr<nova::NovaResourceManager> nova_resource_manager{};

   public:
    CLASS_CM(DisplayManager3D)
    void init(Ui::MainWindow &main_window_ui, ApplicationConfig *global_application_config, OperatorProgressStatus *progress_manager);

    GLViewer *getRealtimeViewer() const { return realtime_viewer; };
    GLViewer *getNovaViewer() const { return nova_viewer; };
    void setNewScene(SceneChangeData scene_change_data);
    void prepareSceneChange();
    SceneTree &getSceneTree() const;
  };
}  // namespace controller
#endif  // DISPLAYMANAGER3D_H
