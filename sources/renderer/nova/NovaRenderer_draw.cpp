#include "CameraFrameBuffer.h"
#include "Config.h"
#include "DrawEngine.h"
#include "Drawable.h"
#include "EnvmapTextureManager.h"
#include "GLMutablePixelBufferObject.h"
#include "NovaRenderer.h"
#include "manager/NovaResourceManager.h"

#include <sys/param.h>

static constexpr int MAX_RECUR_DEPTH = 20;
static constexpr int MAX_SAMPLES = 10000;
static constexpr int NUM_TILES = 20;

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
  nova_resource_manager->envmapSetData(
      &current_envmap->data, current_envmap->metadata.width, current_envmap->metadata.height, current_envmap->metadata.channels);
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
  nova_resource_manager->getEngineData().setTilesWidth(NUM_TILES);
  nova_resource_manager->getEngineData().setTilesHeight(NUM_TILES);
  nova_resource_manager->getEngineData().setAliasingSamples(8);
  nova_resource_manager->getEngineData().setMaxSamples(MAX_SAMPLES);
  nova_resource_manager->getEngineData().setMaxDepth(MAX_RECUR_DEPTH);
  nova_resource_manager->getEngineData().setCancelPtr(&cancel_render);
}

void NovaRenderer::drawBatch() {
  nova_result_futures = nova::draw(&engine_render_buffers,
                                   screen_size.width,
                                   screen_size.height,
                                   nova_engine.get(),
                                   global_application_config->getThreadPool(),
                                   nova_resource_manager.get());
}

void NovaRenderer::doProgressiveRender() {
  nova_result_futures.clear();
  /* Solves the sum :  MAX_SAMPLES = 1 + 2 + 3 + 4 + ...+ smax .*/
  const float s1 = (-1 - std::sqrt(1.f + 8 * nova_resource_manager->getEngineData().getMaxSamples())) * 0.5f;
  const float s2 = (-1 + std::sqrt(1.f + 8 * nova_resource_manager->getEngineData().getMaxSamples())) * 0.5f;
  const float smax = std::max(s1, s2);
  nova_resource_manager->getEngineData().setMaxDepth(nova_resource_manager->getEngineData().getMaxDepth() < MAX_RECUR_DEPTH ?
                                                         nova_resource_manager->getEngineData().getMaxDepth() + 1 :
                                                         MAX_RECUR_DEPTH);
  nova_resource_manager->getEngineData().setSampleIncrement(current_frame);
  if (current_frame < smax)
    drawBatch();
}

void NovaRenderer::resetToBaseState() {
  current_frame = 1;
  cancel_render = true;
  emptyScheduler();
  emptyAccumBuffer();
  populateNovaSceneResources();
  nova_resource_manager->getEngineData().setSampleIncrement(1);
  nova_resource_manager->getEngineData().setMaxDepth(1);
  cancel_render = false;
}

void NovaRenderer::draw() {
  engine_render_buffers.accumulator_buffer = accumulated_render_buffer.data();
  engine_render_buffers.partial_buffer = partial_render_buffer.data();
  engine_render_buffers.byte_size_buffers = screen_size.height * screen_size.width * sizeof(float) * 4;

  populateNovaSceneResources();
  if (needRedraw) {
    resetToBaseState();

    needRedraw = false;
  }

  doProgressiveRender();

  framebuffer_texture->bindTexture();
  pbo_read->bind();
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);  // TODO : use wrappers
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
  camera_framebuffer->renderFrameBufferMesh();
  displayProgress(current_frame, nova_resource_manager->getEngineData().getMaxSamples());
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

void NovaRenderer::prepareRedraw() {
  if (global_application_config && global_application_config->getThreadPool())
    global_application_config->getThreadPool()->emptyQueue();
  cancel_render = true;
  syncRenderEngineThreads();
  cancel_render = false;
  current_frame = 1;
  emptyBuffers();
}

void NovaRenderer::updateNovaCameraFields() {
  if (!scene_camera)
    return;
  scene_camera->computeViewProjection();
  nova::camera::CameraResourcesHolder &nova_camera_structure = nova_resource_manager->getCameraData();
  nova::scene::SceneTransformations &nova_scene_transformations = nova_resource_manager->getSceneTransformation();
  /* Camera data*/
  nova_camera_structure.setUpVector(scene_camera->getUpVector());
  nova_camera_structure.setProjection(scene_camera->getProjection());
  nova_camera_structure.setInvProjection(glm::inverse(scene_camera->getProjection()));
  /* because we use the top transformation matrix also as a view matrix to simulate panning and camera rotation*/
  nova_camera_structure.setView(scene_camera->getView() * scene_camera->getLocalModelMatrix());
  nova_camera_structure.setInvView(glm::inverse(nova_camera_structure.getView()));
  nova_camera_structure.setPosition(scene_camera->getPosition());
  nova_camera_structure.setDirection(scene_camera->getDirection());
  nova_camera_structure.setScreenWidth((int)screen_size.width);
  nova_camera_structure.setScreenHeight((int)screen_size.height);

  /* Scene root transformations */
  nova_scene_transformations.setTranslation(scene_camera->getSceneTranslationMatrix());
  nova_scene_transformations.setInvTranslation(glm::inverse(scene_camera->getSceneTranslationMatrix()));
  nova_scene_transformations.setRotation(scene_camera->getSceneRotationMatrix());
  nova_scene_transformations.setInvRotation(glm::inverse(scene_camera->getSceneRotationMatrix()));
  nova_scene_transformations.setModel(scene_camera->getLocalModelMatrix());
  nova_scene_transformations.setInvModel(glm::inverse(scene_camera->getLocalModelMatrix()));
  nova_scene_transformations.setPvm(scene_camera->getProjection() * scene_camera->getView() * scene_camera->getLocalModelMatrix());
  nova_scene_transformations.setInvPvm(glm::inverse(nova_scene_transformations.getPvm()));
  nova_scene_transformations.setVm(nova_camera_structure.getView() * nova_scene_transformations.getModel());
  nova_scene_transformations.setInvVm(glm::inverse(nova_scene_transformations.getVm()));
  nova_scene_transformations.setNormalMatrix(glm::mat3(glm::transpose(nova_scene_transformations.getInvModel())));
}