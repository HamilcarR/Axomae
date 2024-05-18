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
#include "NovaGeoPrimitive.h"
#include "PerformanceLogger.h"
#include "RenderPipeline.h"
#include "Scene.h"
#include "TextureProcessing.h"
#include "nova_material.h"
#include "shape/Sphere.h"
#include "shape/Square.h"
#include "shape/Triangle.h"
#include <boost/stacktrace/detail/frame_decl.hpp>
#include <unistd.h>

static constexpr int MAX_RECUR_DEPTH = 50;
static std::mutex mutex;
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
  partial_render_buffer.resize(resolution.width * resolution.height * 4);
  accumulated_render_buffer.resize(resolution.width * resolution.height * 4);
  final_render_buffer.resize(resolution.width * resolution.height * 4);
  pbo_read = std::make_unique<GLMutablePixelBufferObject>(GLMutablePixelBufferObject::UP, partial_render_buffer.size() * sizeof(float));
  nova_engine = std::make_unique<NovaLRengineInterface>();
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

  camera_framebuffer->initializeFrameBuffer();
  framebuffer_texture = camera_framebuffer->getFrameBufferTexturePointer(GLFrameBuffer::COLOR0);
  pbo_read->initializeBuffers();
  global_application_config = app_conf;
  scene_camera->computeProjectionSpace();
  initializeEngine();
}

void NovaRenderer::resetToBaseState() {
  current_frame = 1;
  nova_engine_data->renderer_data.max_depth = 1;
  if (global_application_config && global_application_config->getThreadPool())
    global_application_config->getThreadPool()->emptyQueue();
  emptyAccumBuffer();
  populateNovaSceneResources();
}

void NovaRenderer::syncRenderEngineThreads() {
  if (global_application_config && global_application_config->getThreadPool())
    global_application_config->getThreadPool()->fence();
}

void NovaRenderer::setNewScene(const SceneChangeData &new_scene) {
  namespace nova_material = nova::material;
  namespace nova_primitive = nova::primitive;
  namespace nova_shape = nova::shape;

  resetToBaseState();
  syncRenderEngineThreads();
  nova_engine_data->scene_data.materials_collection.clear();
  nova_engine_data->scene_data.primitives.clear();
  nova_engine_data->scene_data.shapes.clear();
  std::unique_ptr<nova_material::NovaMaterialInterface> mat1 = std::make_unique<nova_material::NovaDielectricMaterial>(glm::vec4(1.f, 1.f, 1.f, 1.f),
                                                                                                                       2.54f);
  nova_engine_data->scene_data.materials_collection.push_back(std::move(mat1));
  auto c1 = nova_engine_data->scene_data.materials_collection[0].get();
  for (const auto &elem : new_scene.mesh_list) {
    const Object3D &geometry = elem->getGeometry();
    for (int i = 0; i < geometry.indices.size(); i += 3) {
      glm::vec3 v1{}, v2{}, v3{};
      // V1
      int idx = geometry.indices[i] * 3;
      v1.x = geometry.vertices[idx];
      v1.y = geometry.vertices[idx + 1];
      v1.z = geometry.vertices[idx + 2];
      // V2
      idx = geometry.indices[i + 1] * 3;
      v2.x = geometry.vertices[idx];
      v2.y = geometry.vertices[idx + 1];
      v2.z = geometry.vertices[idx + 2];

      idx = geometry.indices[i + 2] * 3;
      v3.x = geometry.vertices[idx];
      v3.y = geometry.vertices[idx + 1];
      v3.z = geometry.vertices[idx + 2];

      auto tri = nova::shape::NovaShapeInterface::create<nova_shape::Triangle>(v1, v2, v3);
      nova_engine_data->scene_data.shapes.push_back(std::move(tri));
      auto s1 = nova_engine_data->scene_data.shapes.back().get();
      auto primit = nova::primitive::NovaPrimitiveInterface::create<nova_primitive::NovaGeoPrimitive>(s1, c1);
      nova_engine_data->scene_data.primitives.push_back(std::move(primit));
    }
  }
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
  nova_engine_data->renderer_data.sample_increment = current_frame;
}

void NovaRenderer::copyBufferToPbo(float *pbo_map, int width, int height, int channels) {
  float max = 0.f;
  for (int i = 0; i < width * height * channels; i++) {
    const float old = accumulated_render_buffer[i] / (current_frame + 1);
    const float new_ = partial_render_buffer[i];
    const float pix = old + 0.8f * (new_ - old);
    final_render_buffer[i] = pix;
    pbo_map[i] = pix;
    max = std::max(max, pix);
  }
}

void NovaRenderer::initializeEngine() {
  nova_engine_data->renderer_data.tiles_w = 20;
  nova_engine_data->renderer_data.tiles_h = 20;
  nova_engine_data->renderer_data.aliasing_samples = 8;
  nova_engine_data->renderer_data.renderer_max_samples = 10000;
  nova_engine_data->renderer_data.max_depth = MAX_RECUR_DEPTH;
}

void NovaRenderer::displayProgress(float current, float target) {
  if (current > target)
    return;
  if (!gl_widget)
    return;
  if (!gl_widget->getProgressManager())
    return;
  gl_widget->targetProgress(target);
  gl_widget->setCurrent(current);
  gl_widget->setProgressStatusText("Sample: " + std::to_string((int)current) + "/" + std::to_string((int)target) + " ");
  gl_widget->notifyProgress();
}

void NovaRenderer::draw() {

  engine_render_buffers.accumulator_buffer = accumulated_render_buffer.data();
  engine_render_buffers.partial_buffer = partial_render_buffer.data();
  engine_render_buffers.byte_size_buffers = screen_size.height * screen_size.width * sizeof(float) * 4;
  PerformanceLogger perf;
  perf.startTimer();

  populateNovaSceneResources();
  if (needRedraw) {
    resetToBaseState();
    needRedraw = false;
  }

  nova_result_futures.clear();
  const float s1 = (-1 - std::sqrt(1.f + 8 * nova_engine_data->renderer_data.renderer_max_samples)) * 0.5f;
  const float s2 = (-1 + std::sqrt(1.f + 8 * nova_engine_data->renderer_data.renderer_max_samples)) * 0.5f;
  const float smax = std::max(s1, s2);

  if (current_frame < smax) {
    nova_engine_data->renderer_data.max_depth = nova_engine_data->renderer_data.max_depth < MAX_RECUR_DEPTH ?
                                                    nova_engine_data->renderer_data.max_depth + 1 :
                                                    MAX_RECUR_DEPTH;
    nova_result_futures = nova::draw(&engine_render_buffers,
                                     screen_size.width,
                                     screen_size.height,
                                     nova_engine.get(),
                                     global_application_config->getThreadPool(),
                                     nova_engine_data.get());
  }
  framebuffer_texture->bindTexture();
  pbo_read->bind();
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

  // Set texture filtering mode
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  GL_ERROR_CHECK(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, screen_size.width, screen_size.height, 0, GL_RGBA, GL_FLOAT, nullptr));

  pbo_map_buffer = pbo_read->mapBufferRange<float>(0, screen_size.width * screen_size.height * 4 * sizeof(float), 0);
  if (pbo_map_buffer) {
    copyBufferToPbo(pbo_map_buffer, screen_size.width, screen_size.height, 4);
    if (!pbo_read->unmapBuffer())
      LOG("PBO unmap returned false ", LogLevel::WARNING);
  }

  pbo_read->unbind();
  framebuffer_texture->unbindTexture();

  perf.endTimer();
  perf.print();
  camera_framebuffer->renderFrameBufferMesh();

  displayProgress(current_frame, nova_engine_data->renderer_data.renderer_max_samples);
  current_frame++;
  scanline++;
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

void NovaRenderer::emptyAccumBuffer() {
  std::memset(accumulated_render_buffer.data(), 0, screen_size.width * screen_size.height * 4 * sizeof(float));
}
void NovaRenderer::emptyRenderBuffer() { std::memset(partial_render_buffer.data(), 0, screen_size.width * screen_size.height * 4 * sizeof(float)); }

void NovaRenderer::emptyBuffers() {
  for (unsigned i = 0; i < screen_size.width * screen_size.height * 4; i++) {
    partial_render_buffer[i] = 0.f;
    accumulated_render_buffer[i] = 0.f;
    final_render_buffer[i] = 0.f;
  }
}

void NovaRenderer::prepareRedraw() {
  if (global_application_config)
    global_application_config->getThreadPool()->emptyQueue();
  current_frame = 1;
  emptyBuffers();
}

void NovaRenderer::updateNovaCameraFields() {
  if (!scene_camera)
    return;
  scene_camera->computeViewProjection();
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

  nova_engine_data->camera_data.VM = nova_engine_data->camera_data.V * nova_engine_data->camera_data.M;
  nova_engine_data->camera_data.inv_VM = glm::inverse(nova_engine_data->camera_data.V * nova_engine_data->camera_data.M);

  nova_engine_data->camera_data.N = glm::mat3(glm::transpose(nova_engine_data->camera_data.inv_M));

  nova_engine_data->camera_data.position = scene_camera->getPosition();

  nova_engine_data->camera_data.direction = scene_camera->getDirection();

  nova_engine_data->camera_data.screen_width = screen_size.width;
  nova_engine_data->camera_data.screen_height = screen_size.height;
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

Scene &NovaRenderer::getScene() const { return *scene; }

RenderPipeline &NovaRenderer::getRenderPipeline() const { return *render_pipeline; }

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
