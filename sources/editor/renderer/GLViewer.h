#ifndef GLVIEWER_H
#define GLVIEWER_H

#include "Image.h"
#include "Mesh.h"
#include "RendererCallbacks.h"
#include "RendererInterface.h"
#include "SceneHierarchy.h"
#include "constants.h"
#include "internal/common/image/Rgb.h"

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
class QTimer;
/**
 * @class GLViewer
 * This class implements methods for the drawing process
 */
class GLViewer : public QOpenGLWidget, public controller::IProgressManager {

  Q_OBJECT

  ApplicationConfig *global_application_config{};

 protected:
  std::unique_ptr<IRenderer> renderer; /*<Pointer on the renderer of the scene*/
  bool glew_initialized;               /*<Check if context is initialized*/
  std::unique_ptr<controller::event::Event> widget_input_events;

 private:
  bool render_on_timer{false};
  std::unique_ptr<QTimer> timer;
  bool is_rendering{true};

 public:
  explicit GLViewer(QWidget *parent = nullptr);
  explicit GLViewer(std::unique_ptr<IRenderer> &renderer, QWidget *parent = nullptr);
  ~GLViewer() override;
  GLViewer(const GLViewer &copy) = delete;
  GLViewer &operator=(const GLViewer &copy) = delete;
  GLViewer &operator=(GLViewer &&move) noexcept;
  GLViewer(GLViewer &&move) noexcept;

  virtual void setNewScene(const SceneChangeData &new_scene);
  ax_no_discard RendererInterface &getRenderer() const;
  void setApplicationConfig(ApplicationConfig *app_conf);
  void setRenderer(std::unique_ptr<IRenderer> &renderer);
  void renderOnTimer(int interval);
  void renderOnUpdate();
  ax_no_discard image::ImageHolder<uint8_t> getRenderScreenshotUint8(int width, int height) const;
  ax_no_discard image::ImageHolder<float> getRenderScreenshotFloat(int width, int height) const;
  ax_no_discard image::Rgb getFramebufferColor(int x, int y) const;
  void closeEvent(QCloseEvent *event) override;
  void prepareRendererSceneChange();
  void signalEnvmapChange();
  template<RENDERER_CALLBACK_ENUM callback_id, class... Args>
  void rendererCallback(Args &&...args);

 protected:
  void initializeGL() override;
  void paintGL() override;
  void resizeGL(int width, int height) override;
  ax_no_discard const controller::event::Event *getInputEventsStructure() const;

 private:
  void mouseMoveEvent(QMouseEvent *event) override;
  void mousePressEvent(QMouseEvent *event) override;
  void mouseReleaseEvent(QMouseEvent *event) override;
  void mouseDoubleClickEvent(QMouseEvent *event) override;
  void wheelEvent(QWheelEvent *event) override;
  void showEvent(QShowEvent *event) override;
  void hideEvent(QHideEvent *event) override;
 public slots:
  void onUpdateDrawEvent();
  void onTimerTimeout();
  void syncRenderer();
  void haltRender();
  void resumeRender();
  void currentCtx();
  void doneCtx();
};

/* Replace by an event structure in Controller class and send it to the renderer */
template<RENDERER_CALLBACK_ENUM callback_id, class... Args>
void GLViewer::rendererCallback(Args &&...args) {
  renderer->executeMethod<callback_id>(std::forward<Args>(args)...);
  update();
}
#endif
