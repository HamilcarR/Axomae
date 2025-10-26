#ifndef GUIWINDOW_H
#define GUIWINDOW_H

#include "DisplayManager3D.h"
#include "LightControllerUI.h"
#include "Renderer.h"
#include "SceneListView.h"
#include "SceneSelector.h"
#include "constants.h"
#include "nova/bake.h"
#include "nova/bake_render_data.h"
#include "ui_main_window.h"
#include <QtWidgets/qmainwindow.h>

/**
 * @file GUIWindow.h
 */

class ApplicationConfig;
class QTimer;
class SDL_Surface;

namespace controller::event {
  class Event;
}

namespace nova {
  class NovaResourceManager;
}

namespace gui {
  enum IMAGETYPE : unsigned { GREYSCALE_LUMI = 1, HEIGHT = 2, NMAP = 3, DUDV = 4, ALBEDO = 5, GREYSCALE_AVG = 6, PROJECTED_NMAP = 7, INVALID = 8 };
}

namespace controller {

  // TODO :  delete old memory management code
  class HeapManagement;

  class WorkspaceTracker;
  struct ui_render_options;
  struct engine_misc_options;
  template<typename T>
  struct image_type;

  /* Controls the main editor window , deals with event management between main window and the rest of the app. */
  class Controller final : public QMainWindow {  // TODO : May be a bit monolithic for now, needs better separation and modularization.
    Q_OBJECT

   private:
    Ui::MainWindow main_window_ui;

    SceneListView *renderer_scene_list;
    ResourceDatabaseManager &resource_database;
    std::unique_ptr<LightController> light_controller;
    std::unique_ptr<ProgressStatus> progress_manager;
    std::unique_ptr<ApplicationConfig> global_application_config;
    /* UV editor*/
    MeshListView *uv_editor_mesh_list;
    SceneSelector uv_mesh_selector;
    /* Tracks current workspace (which widgets are displayed)*/
    std::unique_ptr<WorkspaceTracker> current_workspace;
    /* Timer for renderer synchro */
    std::unique_ptr<QTimer> timer;

    axstd::ByteArena memory_pool;
    nova_baker_utils::NovaBakingStructure nova_baking_structure;

    DisplayManager3D display_manager;

   public:
    static HeapManagement *_MemManagement;  // TODO : Useless , Refactor

   public:
    explicit Controller(std::unique_ptr<ApplicationConfig> config, QWidget *parent = nullptr);
    ~Controller() override;
    Controller(const Controller &) = delete;
    Controller(Controller &&) = delete;
    Controller &operator=(const Controller &) = delete;
    Controller &operator=(Controller &&) = delete;

    void setApplicationConfig(std::unique_ptr<ApplicationConfig> configuration);
    ax_no_discard ApplicationConfig *getApplicationConfig() const { return global_application_config.get(); }
    Ui::MainWindow &getUi() { return main_window_ui; }
    static SDL_Surface *copy_surface(SDL_Surface *surface);
    void closeEvent(QCloseEvent *event) override;
    ProgressStatus *getProgress() const { return progress_manager.get(); }
    ax_no_discard std::string spawnSaveFileDialogueWidget();
    void cleanupWindowProcess(QWidget *window);
    void cleanupNova();
    ax_no_discard const nova_baker_utils::NovaBakingStructure &getBakingStructure() const { return nova_baking_structure; }
    ax_no_discard nova_baker_utils::NovaBakingStructure &getBakingStructure() { return nova_baking_structure; }
    void novaStopBake();
    void emptySceneCaches();
    nova::device_shared_caches_t &getSharedCaches() { return display_manager.getSharedCaches(); }
    DisplayManager3D &getDisplayManager() { return display_manager; }
    const DisplayManager3D &getDisplayManager() const { return display_manager; }

   private:
    void connect_all_slots();
    QGraphicsView *get_corresponding_view(gui::IMAGETYPE image);
    SDL_Surface *get_corresponding_session_pointer(gui::IMAGETYPE image);
    bool set_corresponding_session_pointer(image_type<SDL_Surface> *image_type_pointer);
    void display_image(SDL_Surface *surf, gui::IMAGETYPE image, bool save_in_heap);
    void do_nova_render(const ui_render_options &render_options, const engine_misc_options &misc_options);

    /* SLOTS */
   public slots:
    void slot_nova_save_bake(const image::ImageHolder<float> &image);
    void slot_nova_stop_bake();
    void slot_nova_start_bake();

    bool slot_import_image();
    bool slot_import_3DOBJ();
    bool slot_import_envmap();
    bool slot_open_project();
    bool slot_save_project();
    bool slot_save_image();

    bool slot_greyscale_average();
    bool slot_greyscale_luminance();
    void slot_use_scharr();
    void slot_use_prewitt();
    void slot_use_sobel();
    void slot_use_gpgpu(bool checked);
    void slot_use_object_space();
    void slot_use_tangent_space();
    void slot_change_nmap_factor(int factor);
    void slot_change_nmap_attenuation(int atten);
    void slot_compute_dudv();
    void slot_change_dudv_nmap(int factor);
    void slot_cubemap_baking();
    void slot_next_mesh();
    void slot_previous_mesh();
    void slot_project_uv_normals();
    void slot_smooth_edge();
    void slot_sharpen_edge();
    void slot_undo();
    void slot_redo();
    void slot_set_renderer_gamma_value(int gamma);
    void slot_set_renderer_exposure_value(int exposure);
    void slot_reset_renderer_camera();
    void slot_set_renderer_no_post_process();
    void slot_set_renderer_edge_post_process();
    void slot_set_renderer_sharpen_post_process();
    void slot_set_renderer_blurr_post_process();
    void slot_set_rasterizer_point();
    void slot_set_rasterizer_fill();
    void slot_set_rasterizer_wireframe();
    void slot_set_display_boundingbox(bool display);
    void slot_on_closed_spawn_window(QWidget *address);
   protected slots:
    void slot_update_smooth_factor(int factor);
    void slot_select_uv_editor_item();
  };

}  // namespace controller

#endif
