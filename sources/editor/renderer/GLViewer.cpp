#include "GLViewer.h"
#include "Config.h"
#include "DebugGL.h"
#include "EventController.h"
#include "Renderer.h"
#include <QCursor>
#include <QMouseEvent>
#include <QPoint>
#include <QSurfaceFormat>

using namespace axomae;
using EventManager = controller::event::Event;

GLViewer::GLViewer(QWidget *parent) : QOpenGLWidget(parent) {
  QSurfaceFormat format;
  format.setRenderableType(QSurfaceFormat::OpenGL);
  format.setVersion(4, 6);
  format.setProfile(QSurfaceFormat::CoreProfile);
  format.setOption(QSurfaceFormat::DebugContext);
  format.setSwapBehavior(QSurfaceFormat::DoubleBuffer);
  format.setAlphaBufferSize(8);
  format.setSwapInterval(1);
  setFormat(format);
  renderer = std::make_unique<Renderer>(width(), height(), this);
  widget_input_events = std::make_unique<EventManager>();
  glew_initialized = false;
}

GLViewer::~GLViewer() {}

void GLViewer::initializeGL() {
  makeCurrent();
  if (!glew_initialized) {
    glewExperimental = GL_TRUE;
    GLenum err = glewInit();
    LOG("glew initialized!", LogLevel::INFO);
    if (err != GLEW_OK) {
      LOG("failed to initialize glew with error : " + std::string(reinterpret_cast<const char *>(glewGetErrorString(err))), LogLevel::CRITICAL);
      exit(EXIT_FAILURE);
    } else {
      glew_initialized = true;
      if (GLEW_ARB_debug_output) {
        glEnable(GL_DEBUG_OUTPUT);
        glDebugMessageCallback(glDebugCallback, nullptr);
      } else {
        LOG("Debug output extension not supported\n", LogLevel::WARNING);
      }
    }
  }
  renderer->onResize(width(), height());
  renderer->initialize(global_application_config);
  errorCheck(__FILE__, __LINE__);
}

void GLViewer::paintGL() {
  if (renderer->prep_draw()) {
    renderer->setDefaultFrameBufferId(defaultFramebufferObject());
    renderer->draw();
  }
}

void GLViewer::resizeGL(int w, int h) {
  QOpenGLWidget::resizeGL(w, h);
  renderer->onResize(width(), height());
}

const controller::event::Event *GLViewer::getInputEventsStructure() const {
  AX_ASSERT(widget_input_events != nullptr);
  return widget_input_events.get();
}

void GLViewer::mouseMoveEvent(QMouseEvent *event) {
  QOpenGLWidget::mouseMoveEvent(event);
  QPoint p = this->mapFromGlobal(QCursor::pos());
  QRect bounds = this->rect();
  widget_input_events->flag |= EventManager::EVENT_MOUSE_MOVE;
  if (bounds.contains(p)) {
    widget_input_events->mouse_state.prev_pos_x = widget_input_events->mouse_state.pos_x;
    widget_input_events->mouse_state.prev_pos_y = widget_input_events->mouse_state.pos_y;
    widget_input_events->mouse_state.pos_x = p.x();
    widget_input_events->mouse_state.pos_y = p.y();
    renderer->processEvent(widget_input_events.get());
  }
  update();
  widget_input_events->flag &= ~EventManager::EVENT_MOUSE_MOVE;
}

void GLViewer::wheelEvent(QWheelEvent *event) {
  QOpenGLWidget::wheelEvent(event);
  widget_input_events->flag |= EventManager::EVENT_MOUSE_WHEEL;
  widget_input_events->mouse_state.wheel_delta = event->angleDelta().y();
  renderer->processEvent(widget_input_events.get());
  update();
  widget_input_events->flag &= ~EventManager::EVENT_MOUSE_WHEEL;
}

void GLViewer::mousePressEvent(QMouseEvent *event) {
  QOpenGLWidget::mousePressEvent(event);
  switch (event->button()) {
    case Qt::LeftButton:
      widget_input_events->flag |= EventManager::EVENT_MOUSE_L_PRESS;
      break;
    case Qt::RightButton:
      widget_input_events->flag |= EventManager::EVENT_MOUSE_R_PRESS;
      break;
    default:
      break;
  }
  renderer->processEvent(widget_input_events.get());
  update();
}

void GLViewer::mouseReleaseEvent(QMouseEvent *event) {
  QOpenGLWidget::mouseReleaseEvent(event);
  /* Need this to reset the mouse release flags, or EVENT_MOUSE_X_RELEASE will stay on even after the end of the event, which is not what we want*/
  EventManager::TYPE reset_release_event = EventManager::NO_EVENT;
  switch (event->button()) {
    case Qt::LeftButton:
      widget_input_events->flag |= EventManager::EVENT_MOUSE_L_RELEASE;
      widget_input_events->flag &= ~EventManager::EVENT_MOUSE_L_PRESS;
      reset_release_event = EventManager::EVENT_MOUSE_L_RELEASE;
      break;
    case Qt::RightButton:
      widget_input_events->flag |= EventManager::EVENT_MOUSE_R_RELEASE;
      widget_input_events->flag &= ~EventManager::EVENT_MOUSE_R_PRESS;
      reset_release_event = EventManager::EVENT_MOUSE_R_RELEASE;
      break;
    default:
      break;
  }
  renderer->processEvent(widget_input_events.get());
  update();
  widget_input_events->flag &= ~reset_release_event;
}

void GLViewer::mouseDoubleClickEvent(QMouseEvent *event) { QOpenGLWidget::mouseDoubleClickEvent(event); }

void GLViewer::setNewScene(std::pair<std::vector<Mesh *>, SceneTree> &new_scene) {
  makeCurrent();
  renderer->set_new_scene(new_scene);
  doneCurrent();
}

void GLViewer::onUpdateDrawEvent() { update(); }

Renderer &GLViewer::getRenderer() const { return *renderer; }
