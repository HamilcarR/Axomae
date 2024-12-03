#ifndef DISPLAYMANAGER3D_H
#define DISPLAYMANAGER3D_H
#include "aggregate/nova_acceleration.h"
#include "nova/bake_render_data.h"
#include "nova_gpu_utils.h"
#include "ui_main_window.h"
#include <manager/NovaResourceManager.h>

class ApplicationConfig;

namespace controller {
  class OperatorProgressStatus;
  class IProgressManager;

  class DisplayManager3D {

   private:
    /* The 3D model viewer */
    GLViewer *realtime_viewer{};
    /* Raytracing engine display*/
    GLViewer *nova_viewer{};
    std::unique_ptr<nova::NovaResourceManager> nova_resource_manager{};
    nova::device_shared_caches_t shared_caches;
    nova_baker_utils::bake_buffers_storage_t bake_buffers_storage;

   public:
    CLASS_CM(DisplayManager3D)
    void init(Ui::MainWindow &main_window_ui, ApplicationConfig *global_application_config, OperatorProgressStatus *progress_manager);

    ax_no_discard GLViewer *getRealtimeViewer() const { return realtime_viewer; };
    ax_no_discard GLViewer *getNovaViewer() const { return nova_viewer; };
    ax_no_discard SceneTree &getSceneTree() const;
    ax_no_discard const nova::device_shared_caches_t &getSharedCaches() const { return shared_caches; }
    ax_no_discard nova::device_shared_caches_t &getSharedCaches() { return shared_caches; }
    ax_no_discard nova::NovaResourceManager *getNovaResourceManager() const { return nova_resource_manager.get(); };
    void setNewScene(SceneChangeData scene_change_data, ProgressStatus *progress_manager = nullptr);
    void prepareSceneChange();
  };
}  // namespace controller
#endif  // DISPLAYMANAGER3D_H
