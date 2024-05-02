#include "NovaRenderer.h"
#include "ArcballCamera.h"
#include "CameraFrameBuffer.h"
#include "Config.h"
#include "DebugGL.h"
#include "DrawEngine.h"
#include "EnvmapTextureManager.h"
#include "EventController.h"
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
  scene_camera = database::node::store<ArcballCamera>(*resource_database->getNodeDatabase(), true, 90.f, 0.1f, 10000.f, 50.f, &screen_size).object;
  scene = std::make_unique<Scene>(*resource_database);
  envmap_manager = std::make_unique<EnvmapTextureManager>(
      *resource_database, screen_size, default_framebuffer_id, *render_pipeline, nullptr, EnvmapTextureManager::SELECTED);
  nova_engine_data = std::make_unique<nova::NovaResources>();
  nova_render_buffer.resize(resolution.width * resolution.height * 4);
  pbo_read = std::make_unique<GLMutablePixelBufferObject>(GLMutablePixelBufferObject::UP, nova_render_buffer.size() * sizeof(float));
  nova_engine = std::make_unique<NovaLRengineInterface>();
  current_frame = next_frame = 0;
}

NovaRenderer::~NovaRenderer() {
  if (!nova_result_futures.empty())
    LOGS("Completing workers tasks...");
  pbo_read->clean();
}

void NovaRenderer::initializeEngine() {
  nova_engine_data->renderer_data.tiles_w = 5;
  nova_engine_data->renderer_data.tiles_h = 5;
  nova_engine_data->renderer_data.render_samples = 100;
}

void NovaRenderer::initialize(ApplicationConfig *app_conf) {
  camera_framebuffer->initializeFrameBuffer();
  framebuffer_texture = camera_framebuffer->getFrameBufferTexturePointer(GLFrameBuffer::COLOR0);
  pbo_read->initializeBuffers();
  global_application_config = app_conf;
  scene_camera->computeProjectionSpace();
  initializeEngine();
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
  /*Setup envmap */
  image::ImageHolder<float> *current_envmap = envmap_manager->currentMutableEnvmapMetadata();
  nova_engine_data->envmap_data.raw_data = &current_envmap->data;
  nova_engine_data->envmap_data.width = (int)current_envmap->metadata.width;
  nova_engine_data->envmap_data.height = (int)current_envmap->metadata.height;

  scene_camera->computeViewProjection();
}

void NovaRenderer::syncRenderEngineThreads() {
  for (auto &elem : nova_result_futures) {
    if (elem.valid())
      elem.get();
  }
  nova_result_futures.clear();
}

void NovaRenderer::copyBufferToPbo(float *pbo_map, int width, int height, int channels) {
  float max_val = 0.f;
  for (unsigned y = 0; y < height; y++)
    for (unsigned x = 0; x < width; x++) {
      unsigned idx = (y * screen_size.width + x) * channels;
      for (unsigned k = 0; k < channels; k++) {
        pbo_map_buffer[idx + k] = nova_render_buffer[idx + k];
        max_val = std::max(max_val, nova_render_buffer[idx + k]);
      }
    }
  renderer_data.max_channel_color_value = max_val;
}

void NovaRenderer::draw() {
  current_frame = current_frame >= screen_size.height ? 0 : current_frame + 1;
  PerformanceLogger perf;
  perf.startTimer();
  if (!nova_render_buffer.empty()) {
    populateNovaSceneResources();
    nova_result_futures = nova::draw(nova_render_buffer.data(),
                                     screen_size.width,
                                     screen_size.height,
                                     nova_engine.get(),
                                     global_application_config->getThreadPool(),
                                     nova_engine_data.get());
  }
  framebuffer_texture->bindTexture();

  pbo_read->bind();
  GL_ERROR_CHECK(glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, screen_size.width, screen_size.height, GL_RGBA, GL_FLOAT, nullptr));
  pbo_map_buffer = pbo_read->mapBufferRange<float>(0, screen_size.width * screen_size.height * 4 * sizeof(float), 0);
  if (pbo_map_buffer) {
    if (needRedraw) {
      std::memset(pbo_map_buffer, 0, screen_size.width * screen_size.height * 4 * sizeof(float));
      needRedraw = false;
    }
    copyBufferToPbo(pbo_map_buffer, screen_size.width, screen_size.height, 4);
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
  scene_camera->computeViewProjection();
  updateNovaCameraFields();
  needRedraw = true;
  current_frame = 0;
  if (global_application_config)
    global_application_config->getThreadPool()->emptyQueue();

  std::memset(nova_render_buffer.data(), 0, screen_size.width * screen_size.height * 4 * sizeof(float));
  if (camera_framebuffer)
    camera_framebuffer->resize();
}

void NovaRenderer::updateNovaCameraFields() {
  if (!scene_camera)
    return;
  nova_engine_data->camera_data.up_vector = scene_camera->getUpVector();

  nova_engine_data->camera_data.P = scene_camera->getProjection();
  nova_engine_data->camera_data.inv_P = glm::inverse(nova_engine_data->camera_data.P);

  nova_engine_data->camera_data.V = scene_camera->getView();
  nova_engine_data->camera_data.inv_V = glm::inverse(nova_engine_data->camera_data.V);

  nova_engine_data->camera_data.T = scene_camera->getSceneTranslationMatrix();
  nova_engine_data->camera_data.inv_T = glm::inverse(scene_camera->getSceneTranslationMatrix());

  nova_engine_data->camera_data.R = scene_camera->getSceneRotationMatrix();
  nova_engine_data->camera_data.inv_R = glm::inverse(scene_camera->getSceneRotationMatrix());

  nova_engine_data->camera_data.M = scene_camera->getLocalModelMatrix();
  nova_engine_data->camera_data.inv_M = glm::inverse(nova_engine_data->camera_data.M);

  nova_engine_data->camera_data.PVM = nova_engine_data->camera_data.P * nova_engine_data->camera_data.V * nova_engine_data->camera_data.M;
  nova_engine_data->camera_data.inv_PVM = glm::inverse(nova_engine_data->camera_data.PVM);

  nova_engine_data->camera_data.N = glm::mat3(glm::transpose(nova_engine_data->camera_data.inv_M));

  nova_engine_data->camera_data.position = scene_camera->getPosition();

  nova_engine_data->camera_data.direction = scene_camera->getDirection();

  nova_engine_data->camera_data.screen_width = screen_size.width;
  nova_engine_data->camera_data.screen_height = screen_size.height;
}

void NovaRenderer::processEvent(const controller::event::Event *event) {
  if (!event)
    return;
  if (scene_camera) {
    scene_camera->processEvent(event);
    /* Camera setup */
    updateNovaCameraFields();
  }
}

void NovaRenderer::getScreenPixelColor(int x, int y, float r_screen_pixel_color[4]) {
  int idx = ((screen_size.height - y) * screen_size.width + x) * 4;
  float r = nova_render_buffer[idx];
  float g = nova_render_buffer[idx + 1];
  float b = nova_render_buffer[idx + 2];

  r_screen_pixel_color[0] = r;
  r_screen_pixel_color[1] = g;
  r_screen_pixel_color[2] = b;
  r_screen_pixel_color[3] = 1.f;
}

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
