#include "NovaRenderer.h"
#include "CameraFrameBuffer.h"
#include "EnvmapTextureManager.h"
#include "RenderPipeline.h"
#include "Scene.h"
NovaRenderer::NovaRenderer(unsigned int width, unsigned int height, GLViewer *widget) : NovaRenderer() {
  resource_database = &ResourceDatabaseManager::getInstance();
  camera_framebuffer = std::make_unique<CameraFrameBuffer>(*resource_database, &screen_size, &default_framebuffer_id);
}

NovaRenderer::~NovaRenderer() = default;

void NovaRenderer::initialize(ApplicationConfig *app_conf) { camera_framebuffer->initializeFrameBuffer(); }
bool NovaRenderer::prep_draw() {
  if (camera_framebuffer && camera_framebuffer->getDrawable()->ready()) {
    camera_framebuffer->startDraw();
    return true;
  } else
    glClearColor(1.f, 1.f, 1.f, 1.f);
}

void NovaRenderer::draw() {
  glClear(GL_COLOR_BUFFER_BIT);
  Texture *tex = camera_framebuffer->getFrameBufferTexturePointer(GLFrameBuffer::COLOR0);
  tex->bindTexture();
  auto result = ResourceDatabaseManager::getInstance().getHdrDatabase()->get(1);
  if (result) {
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, result->metadata().width, result->metadata().height, 0, GL_RGB, GL_FLOAT, result->data.data());
  }
  camera_framebuffer->renderFrameBufferMesh();
}

void NovaRenderer::processEvent(const controller::event::Event *event) const {}
void NovaRenderer::setDefaultFrameBufferId(unsigned int id) {}
unsigned int *NovaRenderer::getDefaultFrameBufferIdPointer() { return nullptr; }
Scene &NovaRenderer::getScene() const {}
RenderPipeline &NovaRenderer::getRenderPipeline() const {}

void NovaRenderer::setNewScene(const SceneChangeData &new_scene) {}

void NovaRenderer::onResize(unsigned int width, unsigned int height) {
  screen_size.width = width;
  screen_size.height = height;
  if (camera_framebuffer) {
    camera_framebuffer->resize();
  }
}

void NovaRenderer::setGammaValue(float value) {
  if (camera_framebuffer) {
    camera_framebuffer->setGamma(value);
  }
}

void NovaRenderer::setExposureValue(float value) {
  if (camera_framebuffer) {
    camera_framebuffer->setExposure(value);
  }
}

void NovaRenderer::setNoPostProcess() {
  if (camera_framebuffer) {
    camera_framebuffer->setPostProcessDefault();
  }
}

void NovaRenderer::setPostProcessEdge() {
  if (camera_framebuffer) {
    camera_framebuffer->setPostProcessEdge();
  }
}

void NovaRenderer::setPostProcessSharpen() {
  if (camera_framebuffer) {
    camera_framebuffer->setPostProcessSharpen();
  }
}

void NovaRenderer::setPostProcessBlurr() {
  if (camera_framebuffer) {
    camera_framebuffer->setPostProcessBlurr();
  }
}

void NovaRenderer::resetSceneCamera() {
  if (scene_camera) {
    scene_camera->reset();
  }
}

void NovaRenderer::setRasterizerFill() {
  if (scene) {
    scene->setPolygonFill();
  }
}

void NovaRenderer::setRasterizerPoint() {
  if (scene) {
    scene->setPolygonPoint();
  }
}

void NovaRenderer::setRasterizerWireframe() {
  if (scene) {
    scene->setPolygonWireframe();
  }
}

void NovaRenderer::displayBoundingBoxes(bool display) {
  if (scene) {
    scene->displayBoundingBoxes(display);
  }
}
void NovaRenderer::setViewerWidget(GLViewer *widget) {
  gl_widget = widget;
  if (render_pipeline)
    render_pipeline->setContextSwitcher(widget);
}
