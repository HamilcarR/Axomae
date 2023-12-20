#include "Renderer.h"
#include "DebugGL.h"
#include "INodeFactory.h"
#include "Loader.h"
#include "RenderPipeline.h"
using namespace axomae;

Renderer::Renderer() : resource_database(ResourceDatabaseManager::getInstance()), scene(resource_database) {
  start_draw = false;
  camera_framebuffer = nullptr;
  mouse_state.pos_x = 0;
  mouse_state.pos_y = 0;
  mouse_state.left_button_clicked = false;
  mouse_state.left_button_released = true;
  mouse_state.right_button_clicked = false;
  mouse_state.right_button_released = true;
  mouse_state.previous_pos_x = 0;
  mouse_state.previous_pos_y = 0;
  default_framebuffer_id = 0;
  Loader loader;
  loader.loadShaderDatabase();
  scene_camera =
      NodeBuilder::store<ArcballCamera>(resource_database.getNodeDatabase(), true, 45.f, &screen_size, 0.1f, 10000.f, 100.f, &mouse_state).object;
}

Renderer::Renderer(unsigned width, unsigned height, GLViewer *widget) : Renderer() {
  screen_size.width = width;
  screen_size.height = height;
  gl_widget = widget;
}

Renderer::~Renderer() {
  if (camera_framebuffer)
    camera_framebuffer->clean();
  scene.clear();
  render_pipeline->clean();
  resource_database.purge();
  light_database.clearDatabase();
  scene_camera = nullptr;
}

void Renderer::initialize() {
  glEnable(GL_DEPTH_TEST);
  glEnable(GL_TEXTURE_CUBE_MAP_SEAMLESS);
  resource_database.getShaderDatabase().initializeShaders();
  camera_framebuffer = std::make_unique<CameraFrameBuffer>(resource_database, &screen_size, &default_framebuffer_id);
  render_pipeline = std::make_unique<RenderPipeline>(this, &resource_database);
  camera_framebuffer->initializeFrameBuffer();
}

bool Renderer::scene_ready() {
  if (!scene.isReady())
    return false;
  if (camera_framebuffer && !camera_framebuffer->getDrawable()->ready())
    return false;
  return true;
}

/*This function is executed each frame*/
bool Renderer::prep_draw() {
  if (start_draw && scene_ready()) {
    camera_framebuffer->startDraw();
    scene.prepare_draw(scene_camera);
    return true;
  } else {
    glClearColor(0, 0, 0, 1.f);
    return false;
  }
}
void Renderer::draw() {
  scene.updateTree();
  camera_framebuffer->bindFrameBuffer();
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  scene.drawForwardTransparencyMode();
  scene.drawBoundingBoxes();
  camera_framebuffer->unbindFrameBuffer();
  camera_framebuffer->renderFrameBufferMesh();
  errorCheck(__FILE__, __LINE__);
}

void Renderer::set_new_scene(std::pair<std::vector<Mesh *>, SceneTree> &new_scene) {
  scene.clear();
  Loader loader;
  EnvironmentMap2DTexture *env = loader.loadHdrEnvmap();  //! TODO in case we want to seek the cubemap to replace it's
                                                          //! texture with this , use visitor pattern in scene graph
  CubeMapMesh *cubemap_mesh = render_pipeline->bakeEnvmapToCubemap(env, 2048, 2048, gl_widget);
  int cube_envmap_id = cubemap_mesh->material.getTextureGroup().getTextureCollection()[0];
  int irradiance_tex_id = render_pipeline->bakeIrradianceCubemap(cube_envmap_id, 64, 64, gl_widget);
  int prefiltered_cubemap = render_pipeline->preFilterEnvmap(cube_envmap_id, 2048, 512, 512, 10, 500, 2, gl_widget);
  int brdf_lut = render_pipeline->generateBRDFLookupTexture(512, 512, gl_widget);
  std::for_each(new_scene.first.begin(),
                new_scene.first.end(),
                [irradiance_tex_id, brdf_lut, prefiltered_cubemap, cube_envmap_id, cubemap_mesh, this](Mesh *m) {
                  m->material.addTexture(irradiance_tex_id);
                  m->material.addTexture(prefiltered_cubemap);
                  m->material.addTexture(brdf_lut);
                  m->setCubemapPointer(cubemap_mesh);
                });
  assert(cubemap_mesh);
  new_scene.first.push_back(cubemap_mesh);
  new_scene.second.pushNewRoot(cubemap_mesh);
  scene.setScene(new_scene);
  scene.setLightDatabasePointer(&light_database);
  scene.setCameraPointer(scene_camera);
  light_database.clearDatabase();
  scene.updateTree();
  scene.generateBoundingBoxes(resource_database.getShaderDatabase().get(Shader::BOUNDING_BOX));
  start_draw = true;
  resource_database.getShaderDatabase().initializeShaders();
  camera_framebuffer->updateFrameBufferShader();
}

void Renderer::onLeftClick() {
  if (event_callback_stack[ON_LEFT_CLICK].empty())
    scene_camera->onLeftClick();
}

void Renderer::onRightClick() { scene_camera->onRightClick(); }

void Renderer::onLeftClickRelease() {
  if (!event_callback_stack[ON_LEFT_CLICK].empty()) {
    const auto to_process = event_callback_stack[ON_LEFT_CLICK].front();
    if (to_process.first == ADD_ELEMENT_POINTLIGHT) {  // TODO : pack this in a class
      glm::mat4 inv_v = glm::inverse(scene_camera->getView());
      glm::mat4 inv_p = glm::inverse(scene_camera->getProjection());
      glm::vec4 w_space = glm::vec4(((float)mouse_state.pos_x * 2.f / (float)screen_size.width) - 1.f,
                                    1.f - (float)mouse_state.pos_y * 2.f / (float)screen_size.height,
                                    1.f,
                                    1.f);

      LightData data = std::any_cast<LightData>(to_process.second);
      w_space = inv_p * w_space;
      w_space = inv_v * w_space;
      glm::vec3 position = w_space / w_space.w;
      position.z = 0.f;
      data.position = glm::inverse(scene_camera->getSceneRotationMatrix()) * glm::vec4(position, 1.f);
      LOG(std::string("x:") + std::to_string(data.position.x) + std::string("  y:") + std::to_string(data.position.y) + std::string("  z:") +
              std::to_string(data.position.z),
          LogLevel::INFO);
      NodeBuilder::store<PointLight>(resource_database.getNodeDatabase(), false, data);
      event_callback_stack[ON_LEFT_CLICK].pop();
      emit sceneModified();
    }
  } else
    scene_camera->onLeftClickRelease();
}

void Renderer::onRightClickRelease() { scene_camera->onRightClickRelease(); }

void Renderer::onScrollDown() { scene_camera->zoomOut(); }

void Renderer::onScrollUp() { scene_camera->zoomIn(); }

void Renderer::onResize(unsigned int width, unsigned int height) {
  screen_size.width = width;
  screen_size.height = height;
  if (camera_framebuffer)
    camera_framebuffer->resize();
}

void Renderer::setGammaValue(float value) {
  if (camera_framebuffer != nullptr) {
    camera_framebuffer->setGamma(value);
    gl_widget->update();
  }
}

void Renderer::setExposureValue(float value) {
  if (camera_framebuffer != nullptr) {
    camera_framebuffer->setExposure(value);
    gl_widget->update();
  }
}

void Renderer::setNoPostProcess() {
  if (camera_framebuffer != nullptr) {
    camera_framebuffer->setPostProcessDefault();
    gl_widget->update();
  }
}

void Renderer::setPostProcessEdge() {
  if (camera_framebuffer != nullptr) {
    camera_framebuffer->setPostProcessEdge();
    gl_widget->update();
  }
}

void Renderer::setPostProcessSharpen() {
  if (camera_framebuffer != nullptr) {
    camera_framebuffer->setPostProcessSharpen();
    gl_widget->update();
  }
}

void Renderer::setPostProcessBlurr() {
  if (camera_framebuffer != nullptr) {
    camera_framebuffer->setPostProcessBlurr();
    gl_widget->update();
  }
}

void Renderer::resetSceneCamera() {
  if (scene_camera != nullptr) {
    scene_camera->reset();
    gl_widget->update();
  }
}

void Renderer::setRasterizerFill() {
  scene.setPolygonFill();
  gl_widget->update();
}

void Renderer::setRasterizerPoint() {
  scene.setPolygonPoint();
  gl_widget->update();
}

void Renderer::setRasterizerWireframe() {
  scene.setPolygonWireframe();
  gl_widget->update();
}

void Renderer::displayBoundingBoxes(bool display) {
  scene.displayBoundingBoxes(display);
  gl_widget->update();
}
