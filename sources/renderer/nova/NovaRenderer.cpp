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
#include "RenderPipeline.h"
#include "Scene.h"
#include "TextureProcessing.h"
#include "manager/NovaResourceManager.h"
#include "material/nova_material.h"
#include "primitive/NovaGeoPrimitive.h"
#include <unistd.h>

static std::mutex mutex;
NovaRenderer::NovaRenderer(unsigned int width, unsigned int height, GLViewer *widget) : NovaRenderer() {
  resource_database = &ResourceDatabaseManager::getInstance();
  camera_framebuffer = std::make_unique<CameraFrameBuffer>(*resource_database, &screen_size, &default_framebuffer_id);
  gl_widget = widget;
  render_pipeline = std::make_unique<RenderPipeline>(&default_framebuffer_id, gl_widget, resource_database);
  scene_camera = database::node::store<ArcballCamera>(*resource_database->getNodeDatabase(), true, 45.f, 0.1f, 10000.f, 50.f, &screen_size).object;
  scene = std::make_unique<Scene>(*resource_database);
  envmap_manager = std::make_unique<EnvmapTextureManager>(
      *resource_database, screen_size, default_framebuffer_id, *render_pipeline, nullptr, EnvmapTextureManager::SELECTED);
  nova_resource_manager = std::make_unique<nova::NovaResourceManager>();
  partial_render_buffer.resize(resolution.width * resolution.height * 4);
  accumulated_render_buffer.resize(resolution.width * resolution.height * 4);
  final_render_buffer.resize(resolution.width * resolution.height * 4);
  pbo_read = std::make_unique<GLMutablePixelBufferObject>(GLMutablePixelBufferObject::UP, partial_render_buffer.size() * sizeof(float));
  nova_engine = std::make_unique<nova::NovaRenderEngineLR>();
  current_frame = next_frame = 0;
}

NovaRenderer::~NovaRenderer() {
  if (!nova_result_futures.empty())
    LOGS("Completing workers tasks...");
  pbo_read->clean();
}

void NovaRenderer::initialize(ApplicationConfig *app_conf) {
  namespace nova_material = nova::material;
  namespace nova_primitive = nova::primitive;
  namespace nova_shape = nova::shape;

  camera_framebuffer->initialize();
  framebuffer_texture = camera_framebuffer->getFrameBufferTexturePointer(GLFrameBuffer::COLOR0);
  pbo_read->initialize();
  global_application_config = app_conf;
  scene_camera->computeProjectionSpace();
  initializeEngine();
}

void NovaRenderer::emptyScheduler() {
  if (global_application_config && global_application_config->getThreadPool()) {
    global_application_config->getThreadPool()->emptyQueue(nova_resource_manager->getEngineData().getTag());
    syncRenderEngineThreads();
  }
}

void NovaRenderer::setProgressStatus(const std::string &status) {
  gl_widget->setProgressStatusText(status);
  gl_widget->notifyProgress();
}
void NovaRenderer::onHideEvent() { onClose(); }

void NovaRenderer::displayProgress(float current, float target) {
  if (current > target)
    return;
  if (!gl_widget)
    return;
  if (!gl_widget->getProgressManager())
    return;
  gl_widget->targetProgress(target);
  gl_widget->setCurrent(current);
  setProgressStatus("Sample: " + std::to_string((int)current) + "/" + std::to_string((int)target) + " ");
}

void NovaRenderer::onResize(unsigned int width, unsigned int height) {
  screen_size.width = width;
  screen_size.height = height;
  updateNovaCameraFields();
  needRedraw = true;
  prepareRedraw();
  if (camera_framebuffer)
    camera_framebuffer->resize();
}
void NovaRenderer::onClose() {
  cancel_render = true;
  emptyScheduler();
}

void NovaRenderer::processEvent(const controller::event::Event *event) {
  using ev = controller::event::Event;
  if (!event)
    return;
  if (scene_camera) {
    scene_camera->processEvent(event);
    /* Camera setup */
    updateNovaCameraFields();
  }
  if (event->flag &
      (ev::EVENT_MOUSE_WHEEL | ev::EVENT_MOUSE_L_PRESS | ev::EVENT_MOUSE_R_PRESS | ev::EVENT_MOUSE_R_RELEASE | ev::EVENT_MOUSE_L_RELEASE))
  {
    needRedraw = true;
  }
}

void NovaRenderer::getScreenPixelColor(int x, int y, float r_screen_pixel_color[4]) {
  if (x >= screen_size.width || x < 0 || y < 0 || y >= screen_size.height)
    return;
  int idx = (((screen_size.height - 1) - y) * screen_size.width + x) * 4;
  float r = accumulated_render_buffer[idx] / current_frame;
  float g = accumulated_render_buffer[idx + 1] / current_frame;
  float b = accumulated_render_buffer[idx + 2] / current_frame;

  r_screen_pixel_color[0] = r;
  r_screen_pixel_color[1] = g;
  r_screen_pixel_color[2] = b;
  r_screen_pixel_color[3] = 1.f;
}

[[nodiscard]] image::ImageHolder<float> NovaRenderer::getSnapshotFloat(int width, int height) const {
  image::ImageHolder<float> img;
  img.data.resize(width * height * 4);
  img.metadata.channels = 4;
  img.metadata.color_corrected = true;
  img.metadata.format = "hdr";
  img.metadata.width = width;
  img.metadata.height = height;
  img.metadata.is_hdr = true;
  TextureOperations<float> op(final_render_buffer, screen_size.width, screen_size.height, 4);
  auto apply_color_correct = [](const float &channel_component) { return channel_component; };
  op.processTexture(img.data.data(), width, height, apply_color_correct);
  return img;
}
[[nodiscard]] image::ImageHolder<uint8_t> NovaRenderer::getSnapshotUint8(int width, int height) const { AX_UNREACHABLE; }

void NovaRenderer::setDefaultFrameBufferId(unsigned int id) {}

unsigned int *NovaRenderer::getDefaultFrameBufferIdPointer() { return nullptr; }

RenderPipeline &NovaRenderer::getRenderPipeline() const { return *render_pipeline; }
