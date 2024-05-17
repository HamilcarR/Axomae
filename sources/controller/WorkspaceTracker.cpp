#include "WorkspaceTracker.h"

uint64_t controller::WorkspaceTracker::getContext() const {
  if (!main_window)
    return 0;
  QString current_tab = main_window->workspace->currentWidget()->objectName();
  uint64_t flag = 0;
  if (current_tab == "rt_renderer") {
    flag |= UI_RENDERER_NOVA;
    flag |= UI_OVERLAY_RGB_DISPLAYER;
  }
  if (current_tab == "gl_renderer")
    flag |= UI_RENDERER_RASTER;

  if (current_tab == "uv_editor")
    flag |= UI_EDITOR_UV;

  if (current_tab == "hdr_editor")
    flag |= UI_EDITOR_BAKER_ENVMAP;

  if (current_tab == "nmap_editor")
    flag |= UI_EDITOR_BAKER_NMAP;

  return flag;
}
