#ifndef GUIWINDOW_H
#define GUIWINDOW_H

#include "BakeRenderData.h"
#include "LightControllerUI.h"
#include "Renderer.h"
#include "SceneListView.h"
#include "SceneSelector.h"
#include "constants.h"
#include "ui_main_window.h"
#include <QtWidgets/qmainwindow.h>
#include <SDL2/SDL_surface.h>

/**
 * @file GUIWindow.h
 */

class ApplicationConfig;
class QTimer;

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

  template<typename T>
  struct image_type;

  class WorkspaceTracker;

  /* Controls the main editor window , deals with event management between main window and the rest of the app */
  class Controller final : public QMainWindow {
    Q_OBJECT

   private:
    Ui::MainWindow main_window_ui;

    /* The 3D model viewer */
    GLViewer *realtime_viewer;
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
    /* Raytracing engine display*/
    GLViewer *nova_viewer;
    NovaBakingStructure nova_baking_structure;

   public:
    static HeapManagement *_MemManagement;  // TODO : Refactor

   public:
    explicit Controller(QWidget *parent = nullptr);
    ~Controller() override;
    void setApplicationConfig(ApplicationConfig &&configuration);
    Ui::MainWindow &getUi() { return main_window_ui; }
    static SDL_Surface *copy_surface(SDL_Surface *surface);
    void closeEvent(QCloseEvent *event) override;
    ProgressStatus *getProgress() const { return progress_manager.get(); }
    [[nodiscard]] std::string spawnSaveFileDialogueWidget();
    void cleanupWindowProcess(QWidget *window);
    void cleanupNova();
    [[nodiscard]] const NovaBakingStructure &getBakingStructure() const { return nova_baking_structure; }
    [[nodiscard]] NovaBakingStructure &getBakingStructure() { return nova_baking_structure; }

   private:
    void connect_all_slots();
    QGraphicsView *get_corresponding_view(gui::IMAGETYPE image);
    SDL_Surface *get_corresponding_session_pointer(gui::IMAGETYPE image);
    bool set_corresponding_session_pointer(image_type<SDL_Surface> *image_type_pointer);
    void display_image(SDL_Surface *surf, gui::IMAGETYPE image, bool save_in_heap);
    void save_bake(const image::ImageHolder<float> &image);

    /* SLOTS */
   public slots:
    bool import_image();
    bool import_3DOBJ();
    bool import_envmap();
    bool open_project();
    bool save_project();
    bool save_image();
    bool greyscale_average();
    bool greyscale_luminance();
    void use_scharr();
    void use_prewitt();
    void use_sobel();
    void use_gpgpu(bool checked);
    void use_object_space();
    void use_tangent_space();
    void change_nmap_factor(int factor);
    void change_nmap_attenuation(int atten);
    void compute_dudv();
    void change_dudv_nmap(int factor);
    void cubemap_baking();
    void nova_baking();
    void next_mesh();
    void previous_mesh();
    void project_uv_normals();
    void smooth_edge();
    void sharpen_edge();
    void undo();
    void redo();
    void set_renderer_gamma_value(int gamma);
    void set_renderer_exposure_value(int exposure);
    void reset_renderer_camera();
    void set_renderer_no_post_process();
    void set_renderer_edge_post_process();
    void set_renderer_sharpen_post_process();
    void set_renderer_blurr_post_process();
    void set_rasterizer_point();
    void set_rasterizer_fill();
    void set_rasterizer_wireframe();
    void set_display_boundingbox(bool display);
    void onClosedSpawnWindow(QWidget *address);
   protected slots:
    void update_smooth_factor(int factor);
    void select_uv_editor_item();
  };

}  // namespace controller

#endif
