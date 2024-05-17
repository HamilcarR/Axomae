#ifndef WORKSPACETRACKER_H
#define WORKSPACETRACKER_H
#include "ui_main_window.h"

namespace controller {
  enum WORKSPACE_WIDGET : uint64_t {
    /*Renderers*/
    UI_RENDERER_RASTER = 1,
    UI_RENDERER_NOVA = 1 << 1,

    /* UV */
    UI_EDITOR_UV = 1 << 10,
    UI_EDITOR_UV_MESH_LIST = 1 << 11,

    /* Bakers */
    UI_EDITOR_BAKER_NMAP = 1 << 20,
    UI_EDITOR_BAKER_ENVMAP = 1 << 21,

    /*Progress bars*/
    UI_EDITOR_BOTTOM_PROGRESS = 1 << 30,

    /*Overlays*/
    UI_OVERLAY_RGB_DISPLAYER = 1ULL << 40,

  };

  class WorkspaceTracker {
   private:
    const Ui::MainWindow *main_window;

   public:
    explicit WorkspaceTracker(const Ui::MainWindow *ui_main_win) : main_window(ui_main_win) {}

    [[nodiscard]] uint64_t getContext() const;
  };
}  // namespace controller
#endif  // WORKSPACETRACKER_H
