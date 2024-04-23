#include "NovaRenderer.h"
#include "CameraFrameBuffer.h"
#include "Config.h"
#include "DebugGL.h"
#include "DrawEngine.h"
#include "EnvmapTextureManager.h"
#include "GLMutablePixelBufferObject.h"
#include "GLViewer.h"
#include "NovaRenderer.h"
#include "PerformanceLogger.h"
#include "RenderPipeline.h"
#include "Scene.h"
#include "TextureProcessing.h"
#include <boost/stacktrace/detail/frame_decl.hpp>
#include <unistd.h>

NovaRenderer::NovaRenderer(unsigned int width, unsigned int height, GLViewer *widget) : NovaRenderer() {
  resource_database = &ResourceDatabaseManager::getInstance();
  camera_framebuffer = std::make_unique<CameraFrameBuffer>(*resource_database, &screen_size, &default_framebuffer_id);
  gl_widget = widget;
  render_pipeline = std::make_unique<RenderPipeline>(&default_framebuffer_id, gl_widget, resource_database);
  scene = std::make_unique<Scene>(*resource_database);
  envmap_manager = std::make_unique<EnvmapTextureManager>(
      *resource_database, screen_size, default_framebuffer_id, *render_pipeline, nullptr, EnvmapTextureManager::SELECTED);
  nova_scene_resources = std::make_unique<nova::NovaResourceHolder>();
  nova_render_buffer.resize(resolution.width * resolution.height * 4);
  pbo_read = std::make_unique<GLMutablePixelBufferObject>(GLMutablePixelBufferObject::UP, nova_render_buffer.size() * sizeof(float));
  current_frame = next_frame = 0;
}

NovaRenderer::~NovaRenderer() {
  if (!nova_result_futures.empty())
    LOGS("Completing workers tasks...");
  pbo_read->clean();
}

void NovaRenderer::initialize(ApplicationConfig *app_conf) {
  camera_framebuffer->initializeFrameBuffer();
  framebuffer_texture = camera_framebuffer->getFrameBufferTexturePointer(GLFrameBuffer::COLOR0);
  pbo_read->initializeBuffers();
  global_application_config = app_conf;
}
bool NovaRenderer::prep_draw() {
  if (camera_framebuffer && camera_framebuffer->getDrawable()->ready())
    camera_framebuffer->startDraw();
  else
    glClearColor(0.f, 0.f, 0.4f, 1.f);
  AX_ASSERT(pbo_read->isReady(), "");
  pbo_read->bind();
  pbo_read->fillBuffers();
  return true;
}

void NovaRenderer::populateNovaSceneResources() {
  if (global_application_config) {
    /* Multithreading */
    nova_scene_resources->thread_pool = global_application_config->getThreadPool();
  }
  /*Setup envmap */
  image::ImageHolder<float> *current_envmap = envmap_manager->currentMutableEnvmapMetadata();
  nova_scene_resources->envmap_data.texture_processor = TextureOperations<float>(
      current_envmap->data, (int)current_envmap->metadata.width, (int)current_envmap->metadata.height);
}

void NovaRenderer::syncRenderEngineThreads() {
  for (auto &elem : nova_result_futures) {
    if (elem.valid())
      elem.get();
  }
  nova_result_futures.clear();
}

void NovaRenderer::draw() {
  current_frame = current_frame >= screen_size.height ? 0 : current_frame + 1;
  PerformanceLogger perf;
  perf.startTimer();
  if (!nova_render_buffer.empty()) {
    populateNovaSceneResources();
    nova_result_futures = nova::draw(nova_render_buffer.data(), screen_size.width, screen_size.height, nova_scene_resources.get());
  }
  framebuffer_texture->bindTexture();

  pbo_read->bind();
  GL_ERROR_CHECK(glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, screen_size.width, screen_size.height, GL_RGBA, GL_FLOAT, nullptr));
  pbo_map_buffer = pbo_read->mapBufferRange<float>(0, screen_size.width * screen_size.height * 4 * sizeof(float), 0);
  if (pbo_map_buffer) {
    if (isResized) {
      std::memset(pbo_map_buffer, 0, screen_size.width * screen_size.height * 4 * sizeof(float));
      isResized = false;
    }
    for (unsigned i = 0; i < screen_size.width; i++) {
      unsigned j = screen_size.height - current_frame;
      unsigned idx = (j * screen_size.width + i) * 4;
      for (unsigned k = 0; k < 4; k++)
        pbo_map_buffer[idx + k] = nova_render_buffer[idx + k];
    }
    if (!pbo_read->unmapBuffer()) {
      LOG("PBO unmap returned false ", LogLevel::WARNING);
    }
  }

  pbo_read->unbind();
  framebuffer_texture->unbindTexture();
  camera_framebuffer->renderFrameBufferMesh();
  perf.endTimer();
  perf.print();
}

void NovaRenderer::onResize(unsigned int width, unsigned int height) {
  screen_size.width = width;
  screen_size.height = height;
  isResized = true;
  if (global_application_config)
    global_application_config->getThreadPool()->emptyQueue();

  std::memset(nova_render_buffer.data(), 0, screen_size.width * screen_size.height * 4 * sizeof(float));
  if (camera_framebuffer) {
    camera_framebuffer->resize();
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
