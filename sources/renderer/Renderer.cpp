#include "Renderer.h"
#include "DebugGL.h"
#include "EnvmapTextureManager.h"
#include "INodeFactory.h"
#include "Loader.h"
#include "RenderPipeline.h"
using namespace axomae;

Renderer::Renderer() : resource_database(ResourceDatabaseManager::getInstance()), scene(resource_database) {
  start_draw = false;
  camera_framebuffer = nullptr;
  mouse_state.pos_x = 0;
  mouse_state.pos_y = 0;
  mouse_state.busy = false;
  mouse_state.left_button_clicked = false;
  mouse_state.left_button_released = true;
  mouse_state.right_button_clicked = false;
  mouse_state.right_button_released = true;
  mouse_state.previous_pos_x = 0;
  mouse_state.previous_pos_y = 0;
  default_framebuffer_id = 0;
  scene_camera =
      database::node::store<ArcballCamera>(resource_database.getNodeDatabase(), true, 45.f, &screen_size, 0.1f, 10000.f, 100.f, &mouse_state).object;
  skybox_mesh = database::node::store<CubeMapMesh>(resource_database.getNodeDatabase(), true).object;
}

Renderer::Renderer(unsigned width, unsigned height, GLViewer *widget) : Renderer() {
  screen_size.width = width;
  screen_size.height = height;
  gl_widget = widget;
  render_pipeline = std::make_unique<RenderPipeline>(default_framebuffer_id, *gl_widget, &resource_database);
  envmap_manager = std::make_unique<EnvmapTextureManager>(
      resource_database, screen_size, default_framebuffer_id, *render_pipeline, *skybox_mesh, scene);
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

static void load_shader_database(ShaderDatabase &shader_database) {
  database::shader::store<BoundingBoxShader>(shader_database, true);
  database::shader::store<BlinnPhongShader>(shader_database, true);
  database::shader::store<CubemapShader>(shader_database, true);
  database::shader::store<ScreenFramebufferShader>(shader_database, true);
  database::shader::store<BRDFShader>(shader_database, true);
  database::shader::store<EnvmapCubemapBakerShader>(shader_database, true);
  database::shader::store<IrradianceCubemapBakerShader>(shader_database, true);
  database::shader::store<EnvmapPrefilterBakerShader>(shader_database, true);
  database::shader::store<BRDFLookupTableBakerShader>(shader_database, true);
}

void Renderer::initialize() {
  glEnable(GL_DEPTH_TEST);
  glEnable(GL_TEXTURE_CUBE_MAP_SEAMLESS);
  /*Read shader + initialize them*/
  ShaderDatabase &shader_database = resource_database.getShaderDatabase();
  load_shader_database(shader_database);
  resource_database.getShaderDatabase().initializeShaders();
  /*Initialize a reusable lut texture*/
  envmap_manager->initializeLUT();
  camera_framebuffer = std::make_unique<CameraFrameBuffer>(resource_database, &screen_size, &default_framebuffer_id);
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
  assert(skybox_mesh);
  scene.clear();
  TextureDatabase &texture_database = resource_database.getTextureDatabase();
  int brdf_lut = envmap_manager->currentLutId();
  if (!skybox_mesh->material.hasTextures()) {
    int irradiance_tex_id = envmap_manager->currentIrradianceId();
    int prefiltered_cubemap = envmap_manager->currentPrefilterId();
    std::for_each(new_scene.first.begin(), new_scene.first.end(), [irradiance_tex_id, brdf_lut, prefiltered_cubemap, this](Mesh *m) {
      m->material.addTexture(irradiance_tex_id);
      m->material.addTexture(prefiltered_cubemap);
      m->material.addTexture(brdf_lut);
      m->setCubemapPointer(skybox_mesh);
    });

  } else {
    int irradiance_tex_id = envmap_manager->currentIrradianceId();
    int prefiltered_cubemap = envmap_manager->currentPrefilterId();
    std::for_each(new_scene.first.begin(), new_scene.first.end(), [irradiance_tex_id, brdf_lut, prefiltered_cubemap, this](Mesh *m) {
      m->material.addTexture(irradiance_tex_id);
      m->material.addTexture(prefiltered_cubemap);
      m->material.addTexture(brdf_lut);
      m->setCubemapPointer(skybox_mesh);
    });
  }

  new_scene.first.push_back(skybox_mesh);
  new_scene.second.pushNewRoot(skybox_mesh);
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

void Renderer::onRightClick() const { scene_camera->onRightClick(); }

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
      database::node::store<PointLight>(resource_database.getNodeDatabase(), false, data);
      event_callback_stack[ON_LEFT_CLICK].pop();
      emit sceneModified();
    }
  } else
    scene_camera->onLeftClickRelease();
}

void Renderer::onRightClickRelease() const { scene_camera->onRightClickRelease(); }

void Renderer::onScrollDown() const { scene_camera->zoomOut(); }

void Renderer::onScrollUp() const { scene_camera->zoomIn(); }

void Renderer::onResize(unsigned int width, unsigned int height) {
  screen_size.width = width;
  screen_size.height = height;
  if (camera_framebuffer)
    camera_framebuffer->resize();
}

void Renderer::setGammaValue(float value) const {
  if (camera_framebuffer != nullptr) {
    camera_framebuffer->setGamma(value);
    gl_widget->update();
  }
}

void Renderer::setExposureValue(float value) const {
  if (camera_framebuffer != nullptr) {
    camera_framebuffer->setExposure(value);
    gl_widget->update();
  }
}

void Renderer::setNoPostProcess() const {
  if (camera_framebuffer != nullptr) {
    camera_framebuffer->setPostProcessDefault();
    gl_widget->update();
  }
}

void Renderer::setPostProcessEdge() const {
  if (camera_framebuffer != nullptr) {
    camera_framebuffer->setPostProcessEdge();
    gl_widget->update();
  }
}

void Renderer::setPostProcessSharpen() const {
  if (camera_framebuffer != nullptr) {
    camera_framebuffer->setPostProcessSharpen();
    gl_widget->update();
  }
}

void Renderer::setPostProcessBlurr() const {
  if (camera_framebuffer != nullptr) {
    camera_framebuffer->setPostProcessBlurr();
    gl_widget->update();
  }
}

void Renderer::resetSceneCamera() const {
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
