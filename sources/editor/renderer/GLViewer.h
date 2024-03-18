#ifndef GLVIEWER_H
#define GLVIEWER_H

#include "Mesh.h"
#include "RendererEnums.h"
#include "SceneHierarchy.h"
#include "constants.h"
#include "utils_3D.h"
#include <QOpenGLWidget>

/**
 * @file GLView.h
 * This file implements Viewer widget in the QT interface
 */

class Renderer;
class QMouseEvent;
class ApplicationConfig;
namespace controller::event {
  class Event;
}
/**
 * @class GLViewer
 * This class implements methods for the drawing process
 */
class GLViewer : public QOpenGLWidget {
  Q_OBJECT
 public:
  explicit GLViewer(QWidget *parent = nullptr);

  ~GLViewer() override;

  virtual void setNewScene(std::pair<std::vector<Mesh *>, SceneTree> &new_scene);

  [[nodiscard]] const Renderer &getConstRenderer() const;

  [[nodiscard]] Renderer &getRenderer() const;

  void setApplicationConfig(ApplicationConfig *app_conf) { global_application_config = app_conf; }

 protected:
  void initializeGL() override;

  void paintGL() override;

  void resizeGL(int width, int height) override;

  [[nodiscard]] const controller::event::Event *getInputEventsStructure() const;

 private:
  void mouseMoveEvent(QMouseEvent *event) override;

  void mousePressEvent(QMouseEvent *event) override;

  void mouseReleaseEvent(QMouseEvent *event) override;

  void mouseDoubleClickEvent(QMouseEvent *event) override;

  void wheelEvent(QWheelEvent *event) override;

 public slots:
  void onUpdateDrawEvent();

 private:
  std::unique_ptr<Renderer> renderer; /*<Pointer on the renderer of the scene*/
  bool glew_initialized;              /*<Check if context is initialized*/
  ApplicationConfig *global_application_config{};
  std::unique_ptr<controller::event::Event> widget_input_events;
};

#endif
