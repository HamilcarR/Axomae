#include "CameraFrameBuffer.h"
#include "Config.h"
#include "Drawable.h"
#include "EnvmapTextureManager.h"
#include "NovaRenderer.h"
#include "bake.h"
#include <cstring>
#include <internal/device/rendering/opengl/GLMutablePixelBufferObject.h>
#include <internal/macro/project_macros.h>
#include <nova/api_engine.h>

static constexpr int MAX_RECUR_DEPTH = 20;
static constexpr int MAX_SAMPLES = 10000;
static constexpr int NUM_TILES = 20;

const char *const NOVA_REALTIME_TAG = "_NOVAREALTIME_POOL_TAG_";
bool NovaRenderer::prep_draw() {
  if (camera_framebuffer && camera_framebuffer->getDrawable()->ready())
    camera_framebuffer->startDraw();
  else
    glClearColor(0.f, 0.f, 0.4f, 1.f);
  AX_ASSERT(pbo_read->isReady(), "");
  pbo_read->bind();
  pbo_read->fill();
  return true;
}

void NovaRenderer::populateNovaSceneResources() {
  if (!envmap_manager || !nova_resource_manager)
    return;
  std::size_t id = envmap_manager->getEnvmapID();
  nova_resource_manager->getTexturesData().setEnvmapId(id);
}

void NovaRenderer::copyBufferToPbo(float *pbo_map, int width, int height, int channels) {
  float max = 0.f;
  for (std::size_t idx = 0; idx < height * width * channels; idx++) {
    const float old = accumulated_render_buffer[idx] / (float)(current_frame + 1);
    const float new_ = partial_render_buffer[idx];
    const float pix = old + 0.8f * (new_ - old);
    final_render_buffer[idx] = pix;
    pbo_map[idx] = pix;
    max = std::max(max, pix);
  }
}

void NovaRenderer::initializeEngine() {
  nova_baker_utils::engine_data engine_opts;
  engine_opts.aa_samples = 8;
  engine_opts.num_tiles_h = engine_opts.num_tiles_w = NUM_TILES;
  engine_opts.depth_max = MAX_RECUR_DEPTH;
  engine_opts.samples_max = MAX_SAMPLES;
  engine_opts.threadpool_tag = NOVA_REALTIME_TAG;
  engine_opts.flip_v = false;
  engine_opts.engine_type_flag = nova::integrator::COMBINED | nova::integrator::PATH;

  initialize_engine_opts(engine_opts, nova_resource_manager->getEngineData());
}

void NovaRenderer::drawBatch() {
  nova::nova_eng_internals interns{nova_resource_manager, nova_exception_manager.get()};
  nova_result_futures = nova::draw(
      &engine_render_buffers, screen_size.width, screen_size.height, nova_engine.get(), global_application_config->getThreadPool(), interns);
}

void NovaRenderer::doProgressiveRender() {
  nova_result_futures.clear();

  const float smax = math::calculus::compute_serie_term(nova_resource_manager->getEngineData().renderer_max_samples);
  nova_resource_manager->getEngineData().max_depth = nova_resource_manager->getEngineData().max_depth < MAX_RECUR_DEPTH ?
                                                         nova_resource_manager->getEngineData().max_depth + 1 :
                                                         MAX_RECUR_DEPTH;
  nova_resource_manager->getEngineData().frame_index = current_frame;
  if (current_frame < smax)
    drawBatch();
}

void NovaRenderer::resetToBaseState() {
  current_frame = 1;
  nova_resource_manager->getEngineData().is_rendering = false;
  emptyScheduler();
  emptyAccumBuffer();
  populateNovaSceneResources();
  nova_resource_manager->getEngineData().frame_index = 1;
  nova_resource_manager->getEngineData().max_depth = 1;
  nova_resource_manager->getEngineData().is_rendering = true;
}

void NovaRenderer::draw() {
  engine_render_buffers.partial_buffer = partial_render_buffer.data();
  engine_render_buffers.depth_buffer = depth_buffer.data();
  engine_render_buffers.byte_size_color_buffers = screen_size.height * screen_size.width * sizeof(float) * 4;

  populateNovaSceneResources();
  if (needRedraw) {
    resetToBaseState();
    needRedraw = false;
  }

  doProgressiveRender();

  framebuffer_texture->bind();
  pbo_read->bind();
  ax_glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  ax_glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

  // Set texture filtering mode
  ax_glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  ax_glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  ax_glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, screen_size.width, screen_size.height, 0, GL_RGBA, GL_FLOAT, nullptr);

  pbo_map_buffer = pbo_read->mapBufferRange<float>(0, screen_size.width * screen_size.height * 4 * sizeof(float), 0);
  if (pbo_map_buffer) {
    copyBufferToPbo(pbo_map_buffer, screen_size.width, screen_size.height, 4);
    if (!pbo_read->unmapBuffer())
      LOG("PBO unmap returned false ", LogLevel::WARNING);
  }

  pbo_read->unbind();
  framebuffer_texture->unbind();
  camera_framebuffer->renderFrameBufferMesh();
  displayProgress(current_frame, nova_resource_manager->getEngineData().renderer_max_samples);
  current_frame++;
  scanline++;
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

void NovaRenderer::syncRenderEngineThreads() {
  if (global_application_config && global_application_config->getThreadPool())
    global_application_config->getThreadPool()->fence(NOVA_REALTIME_TAG);
}

void NovaRenderer::prepareRedraw() {
  if (global_application_config && global_application_config->getThreadPool())
    global_application_config->getThreadPool()->emptyQueue(NOVA_REALTIME_TAG);
  nova_resource_manager->getEngineData().is_rendering = false;
  syncRenderEngineThreads();
  nova_resource_manager->getEngineData().is_rendering = true;
  current_frame = 1;
  emptyBuffers();
}

void NovaRenderer::updateNovaCameraFields() {
  if (!scene_camera)
    return;
  scene_camera->computeViewProjection();
  nova::camera::CameraResourcesHolder &nova_camera_structure = nova_resource_manager->getCameraData();
  nova::scene::SceneTransformations &nova_scene_transformations = nova_resource_manager->getSceneTransformation();
  nova_baker_utils::camera_data c_data{};
  c_data.view = scene_camera->getTransformedView();
  c_data.projection = scene_camera->getProjection();
  c_data.up_vector = scene_camera->getUpVector();
  c_data.position = scene_camera->getPosition();
  c_data.direction = scene_camera->getDirection();
  nova_baker_utils::scene_transform_data st_data{};
  st_data.root_transformation = scene_camera->getLocalModelMatrix();
  st_data.root_rotation = scene_camera->getSceneRotationMatrix();
  st_data.root_translation = scene_camera->getSceneTranslationMatrix();

  initialize_scene_data(c_data, st_data, nova_scene_transformations, nova_camera_structure);
  nova_camera_structure.screen_width = (int)screen_size.width;
  nova_camera_structure.screen_height = (int)screen_size.height;
}
