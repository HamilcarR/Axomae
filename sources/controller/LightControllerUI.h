#ifndef LIGHTCONTROLLERUI_H
#define LIGHTCONTROLLERUI_H

#include "LightingSystem.h"
#include "ui_main_window.h"

/**
 * @brief Implements an event controller for the lighting panel
 */
class LightController : public QObject {
  Q_OBJECT

 private:
  Ui::MainWindow &ui;
  GLViewer *viewer_3d{};
  SceneListView *scene_list_view{};

 public:
  explicit LightController(Ui::MainWindow &_ui) : ui(_ui) {}
  void connectAllSlots();
  void setup(GLViewer *viewer, SceneListView *list_view) {
    viewer_3d = viewer;
    scene_list_view = list_view;
  }

  template<AbstractLight::TYPE type>
  LightData loadFromUi() const;

 protected slots:
  void slot_add_point_light();
  void slot_delete_point_light();

  void slot_add_directional_light();
  void slot_delete_directional_light();

  void slot_add_spot_light();
  void slot_delete_spot_light();
};

#endif