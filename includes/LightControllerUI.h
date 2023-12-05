#ifndef LIGHTCONTROLLERUI_H
#define LIGHTCONTROLLERUI_H

#include "../Form Files/ui_test.h"

/**
 * @brief
 * @class LightController
 *
 */
class LightController : public QObject {
  Q_OBJECT
 public:
  LightController(Ui::MainWindow &_ui) : ui(_ui) {}
  virtual ~LightController() {}
  void connect_all_slots();
  void setup(GLViewer *viewer, SceneListView *list_view) {
    viewer_3d = viewer;
    scene_list_view = list_view;
  }

  template<AbstractLight::TYPE type>
  LightData loadFromUi() const;

 protected slots:
  void addPointLight();
  void deletePointLight();

  void addDirectionalLight();
  void deleteDirectionalLight();

  void addSpotLight();
  void deleteSpotLight();

 public:
  Ui::MainWindow &ui;
  GLViewer *viewer_3d;

 private:
  SceneListView *scene_list_view;
};

#endif