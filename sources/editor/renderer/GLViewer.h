#ifndef GLVIEWER_H
#define GLVIEWER_H

#include "Mesh.h"
#include "RendererEnums.h"
#include "RendererInterface.h"
#include "SceneHierarchy.h"
#include "constants.h"
#include "utils_3D.h"
#include <QOpenGLWidget>
/**
 * @file GLView.h
 * This file implements Viewer widget in the QT interface
 */

class RendererInterface;
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

 protected:
  std::unique_ptr<IRenderer> renderer; /*<Pointer on the renderer of the scene*/
  bool glew_initialized;               /*<Check if context is initialized*/
  ApplicationConfig *global_application_config{};
  std::unique_ptr<controller::event::Event> widget_input_events;

 public:
  explicit GLViewer(QWidget *parent = nullptr);
  explicit GLViewer(std::unique_ptr<IRenderer> &renderer, QWidget *parent = nullptr);
  ~GLViewer() override;
  GLViewer(const GLViewer &copy) = delete;
  GLViewer &operator=(const GLViewer &copy) = delete;
  GLViewer &operator=(GLViewer &&move) noexcept;
  GLViewer(GLViewer &&move) noexcept;
  virtual void setNewScene(const SceneChangeData &new_scene);
  [[nodiscard]] RendererInterface &getRenderer() const;
  void setApplicationConfig(ApplicationConfig *app_conf) { global_application_config = app_conf; }
  void setRenderer(std::unique_ptr<IRenderer> &renderer);
  template<RENDERER_CALLBACK_ENUM callback_id, class... Args>
  constexpr void rendererCallback(Args &&...args);

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
};

template<RENDERER_CALLBACK_ENUM callback_id, class... Args>
constexpr void GLViewer::rendererCallback(Args &&...args) {
  renderer->executeMethod<callback_id>(std::forward<Args>(args)...);
  update();
}
#endif
