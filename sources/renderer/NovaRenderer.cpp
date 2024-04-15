#include "NovaRenderer.h"
#include "CameraFrameBuffer.h"
#include "DebugGL.h"
#include "DrawEngine.h"
#include "EnvmapTextureManager.h"
#include "GLPixelBufferObject.h"
#include "GLViewer.h"
#include "NovaRenderer.h"
#include "OfflineCubemapProcessing.h"
#include "PerformanceLogger.h"
#include "RenderPipeline.h"
#include "Scene.h"

NovaRenderer::NovaRenderer(unsigned int width, unsigned int height, GLViewer *widget) : NovaRenderer() {
  resource_database = &ResourceDatabaseManager::getInstance();
  camera_framebuffer = std::make_unique<CameraFrameBuffer>(*resource_database, &screen_size, &default_framebuffer_id);
  gl_widget = widget;
  render_pipeline = std::make_unique<RenderPipeline>(&default_framebuffer_id, gl_widget, resource_database);
  pixel_buffer_object = std::make_unique<GLPixelBufferObject>(GLPixelBufferObject::UP, screen_size.width * screen_size.height * 3 * sizeof(float));
  scene = std::make_unique<Scene>(*resource_database);
  envmap_manager = std::make_unique<EnvmapTextureManager>(
      *resource_database, screen_size, default_framebuffer_id, *render_pipeline, nullptr, EnvmapTextureManager::SELECTED);
}

NovaRenderer::~NovaRenderer() { pixel_buffer_object->clean(); }

void NovaRenderer::initialize(ApplicationConfig *app_conf) {
  camera_framebuffer->initializeFrameBuffer();
  framebuffer_texture = camera_framebuffer->getFrameBufferTexturePointer(GLFrameBuffer::COLOR0);
  pixel_buffer_object->initializeBuffers();
}
bool NovaRenderer::prep_draw() {
  if (camera_framebuffer && camera_framebuffer->getDrawable()->ready())
    camera_framebuffer->startDraw();
  if (pixel_buffer_object->isReady()) {
    pixel_buffer_object->bind();
    pixel_buffer_object->fillBuffers();
  }
  glClearColor(1.f, 1.f, 1.f, 1.f);
  return true;
}

void NovaRenderer::draw() {

  glClear(GL_COLOR_BUFFER_BIT);
  framebuffer_texture->bindTexture();
  pixel_buffer_object->bind();
  GL_ERROR_CHECK(glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, screen_size.width, screen_size.height, GL_RGB, GL_FLOAT, nullptr));
  pixel_buffer_object->fillBuffers();
  float *map_buffer = pixel_buffer_object->mapBuffer<float>(GLPixelBufferObject::RW);
  if (map_buffer) {
    PerformanceLogger logger;
    logger.startTimer();
    const image::ImageHolder<float> *current_envmap = envmap_manager->currentEnvmapMetadata();
    nova::texturing::SceneResourcesHolder holder{
        EnvmapProcessing<float>(current_envmap->data, (int)current_envmap->metadata.width, (int)current_envmap->metadata.height)};
    nova::draw(map_buffer, (int)screen_size.width, (int)screen_size.height, &holder);
    logger.endTimer();
    logger.print();
  }
  AX_ASSERT(pixel_buffer_object->unmapBuffer(), "");
  pixel_buffer_object->unbind();
  camera_framebuffer->renderFrameBufferMesh();
}

void NovaRenderer::onResize(unsigned int width, unsigned int height) {
  screen_size.width = width;
  screen_size.height = height;
  if (camera_framebuffer) {
    camera_framebuffer->resize();
  }
  if (pixel_buffer_object && pixel_buffer_object->isReady()) {
    pixel_buffer_object->bind();
    pixel_buffer_object->setNewSize((int)width * (int)height * 3 * sizeof(float));
    pixel_buffer_object->unbind();
  }
}

void NovaRenderer::processEvent(const controller::event::Event *event) {}

void NovaRenderer::setDefaultFrameBufferId(unsigned int id) {}

unsigned int *NovaRenderer::getDefaultFrameBufferIdPointer() { return nullptr; }

Scene &NovaRenderer::getScene() const { return *scene; }

RenderPipeline &NovaRenderer::getRenderPipeline() const { return *render_pipeline; }

void NovaRenderer::setNewScene(const SceneChangeData &new_scene) {}

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
