#ifndef DISPLAYMANAGER3D_H
#define DISPLAYMANAGER3D_H
#include "nova/bake_render_data.h"
#include "ui_main_window.h"
#include <nova/api.h>

class ApplicationConfig;

namespace controller {
  class OperatorProgressStatus;
  class IProgressManager;

  class DisplayManager3D : public QObject {
    Q_OBJECT
   private:
    /* The 3D model viewer */
    GLViewer *realtime_viewer{};
    /* Raytracing engine display*/
    GLViewer *nova_viewer{};
    std::unique_ptr<nova::NovaResourceManager> nova_resource_manager{};
    nova::device_shared_caches_t shared_caches;
    nova_baker_utils::bake_buffers_storage_t bake_buffers_storage;
    // TODO : add EnvmapManager instance here ?
   public:
    void init(Ui::MainWindow &main_window_ui, ApplicationConfig *global_application_config, OperatorProgressStatus *progress_manager);
    ax_no_discard GLViewer *getRealtimeViewer() const { return realtime_viewer; };
    ax_no_discard GLViewer *getNovaViewer() const { return nova_viewer; };
    ax_no_discard SceneTree &getSceneTree() const;
    ax_no_discard const nova::device_shared_caches_t &getSharedCaches() const { return shared_caches; }
    ax_no_discard nova::device_shared_caches_t &getSharedCaches() { return shared_caches; }
    ax_no_discard nova::NovaResourceManager *getNovaResourceManager() const { return nova_resource_manager.get(); };
    void setNewScene(SceneChangeData &scene_change_data, ProgressStatus *progress_manager = nullptr);
    void prepareSceneChange();
    void haltRenderers();
    void resumeRenderers();
    void makeCtxRealtime();
    void doneCtxRealtime();
    void onEnvmapChange();

   private:
    void connect_slots();

   public:
   signals:
    void signal_halt_renderers();
    void signal_resume_renderers();
    void signal_sync_renderers();
    void signal_switch_realtime_ctx();
    void signal_done_realtime_ctx();
  };

}  // namespace controller
#endif  // DISPLAYMANAGER3D_H
