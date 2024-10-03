#include "Renderer.h"
#include "ArcballCamera.h"
#include "Config.h"
#include "EnvmapTextureManager.h"
#include "RenderPipeline.h"
#include "ShaderDatabase.h"
#include "event/EventController.h"
#include "internal/device/rendering/opengl/DebugGL.h"
using namespace axomae;

Renderer::Renderer()
    : camera_framebuffer(nullptr), start_draw(false), resource_database(&ResourceDatabaseManager::getInstance()), default_framebuffer_id(0) {
  INodeDatabase *node_db = resource_database->getNodeDatabase();
  scene_camera = database::node::store<ArcballCamera>(*node_db, true, 45.f, 0.1f, 10000.f, 50.f, &screen_size).object;
  scene = std::make_unique<Scene>(ResourceDatabaseManager::getInstance());
  AX_ASSERT_NOTNULL(scene);
  scene->setLightDatabasePointer(&light_database);
  LightData default_dir_light;
  default_dir_light.direction = glm::vec3(0.f, 100.f, 0.f);
  default_dir_light.asPbrColor(255, 255, 255);
  default_dir_light.intensity = 0.0007f;
  default_dir_light.parent = scene_camera;
  default_dir_light.name = "Default-DirectionalLight";
  auto query = database::node::store<DirectionalLight>(*resource_database->getNodeDatabase(), true, default_dir_light);
  light_database.addLight(query.id);
  camera_framebuffer = std::make_unique<CameraFrameBuffer>(*resource_database, &screen_size, &default_framebuffer_id);
}

Renderer::Renderer(Renderer &&move) noexcept {
  render_pipeline = std::move(move.render_pipeline);
  camera_framebuffer = std::move(move.camera_framebuffer);
  start_draw = move.start_draw;
  resource_database = move.resource_database;
  scene = std::move(move.scene);
  scene_camera = move.scene_camera;
  screen_size = move.screen_size;
  default_framebuffer_id = move.default_framebuffer_id;
  light_database = std::move(light_database);
  gl_widget = move.gl_widget;
  envmap_manager = std::move(move.envmap_manager);
}
Renderer &Renderer::operator=(Renderer &&move) noexcept {
  if (this != &move) {
    render_pipeline = std::move(move.render_pipeline);
    camera_framebuffer = std::move(move.camera_framebuffer);
    start_draw = move.start_draw;
    resource_database = move.resource_database;
    scene = std::move(move.scene);
    scene_camera = move.scene_camera;
    screen_size = move.screen_size;
    default_framebuffer_id = move.default_framebuffer_id;
    light_database = std::move(light_database);
    gl_widget = move.gl_widget;
    envmap_manager = std::move(move.envmap_manager);
  }
  return *this;
}

Renderer::Renderer(unsigned width, unsigned height, GLViewer *widget) : Renderer() {
  screen_size.width = width;
  screen_size.height = height;
  gl_widget = widget;
  render_pipeline = std::make_unique<RenderPipeline>(&default_framebuffer_id, gl_widget, resource_database);
  envmap_manager = std::make_unique<EnvmapTextureManager>(*resource_database, screen_size, default_framebuffer_id, *render_pipeline, scene.get());
}

Renderer::~Renderer() {
  if (camera_framebuffer)
    camera_framebuffer->clean();
  scene->clear();
  render_pipeline->clean();
  resource_database->purge();
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

void Renderer::initialize(ApplicationConfig *app_conf) {
  glEnable(GL_DEPTH_TEST);
  glEnable(GL_TEXTURE_CUBE_MAP_SEAMLESS);
  /*Read shader + initialize them*/
  ShaderDatabase *shader_database = resource_database->getShaderDatabase();
  load_shader_database(*shader_database);
  shader_database->initializeShaders();
  /*Initialize a reusable lut texture*/
  scene->initialize();
  envmap_manager->initializeDefaultEnvmap(app_conf);
  camera_framebuffer->initialize();
}

void Renderer::getScreenPixelColor(int x, int y, float r_screen_pixel_color[4]) { EMPTY_FUNCBODY; }

bool Renderer::scene_ready() {
  if (!scene->isReady())
    return false;
  if (camera_framebuffer && !camera_framebuffer->getDrawable()->ready())
    return false;
  return true;
}

/*This function is executed each frame*/
bool Renderer::prep_draw() {
  if (start_draw && scene_ready()) {
    camera_framebuffer->startDraw();
    scene->prepare_draw(scene_camera);
    return true;
  } else {
    glClearColor(0, 0, 0, 1.f);
    return false;
  }
}
void Renderer::draw() {
  scene->updateTree();
  camera_framebuffer->bind();
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  scene->drawForwardTransparencyMode();
  scene->drawBoundingBoxes();
  camera_framebuffer->unbind();
  camera_framebuffer->renderFrameBufferMesh();
  errorCheck(__FILE__, __LINE__);
}

void Renderer::prepSceneChange() { scene->clear(); }
void Renderer::onHideEvent() { EMPTY_FUNCBODY; }

void Renderer::setNewScene(const SceneChangeData &new_scene) {
  AX_ASSERT(new_scene.scene, "");
  AX_ASSERT(camera_framebuffer, "Screen Framebuffer is not set.");
  scene->setScene(*new_scene.scene, new_scene.mesh_list);
  scene->switchEnvmap(envmap_manager->currentCubemapId(),
                      envmap_manager->currentIrradianceId(),
                      envmap_manager->currentPrefilterId(),
                      envmap_manager->currentLutId());

  scene->setCameraPointer(scene_camera);
  scene->updateTree();
  ShaderDatabase *shader_database = resource_database->getShaderDatabase();
  scene->generateBoundingBoxes(shader_database->get(Shader::BOUNDING_BOX));
  start_draw = true;
  shader_database->initializeShaders();
  camera_framebuffer->updateFrameBufferShader();
}

[[nodiscard]] image::ImageHolder<float> Renderer::getSnapshotFloat(int width, int height) const { AX_UNREACHABLE; }
[[nodiscard]] image::ImageHolder<uint8_t> Renderer::getSnapshotUint8(int width, int height) const { AX_UNREACHABLE; }

void Renderer::processEvent(const controller::event::Event *event) {
  scene->processEvent(event);
  scene_camera->processEvent(event);
}

void Renderer::onResize(unsigned int width, unsigned int height) {
  screen_size.width = width;
  screen_size.height = height;
  if (camera_framebuffer) {
    camera_framebuffer->resize();
  }
}
void Renderer::onClose() {}

void Renderer::setGammaValue(float value) {
  if (camera_framebuffer) {
    camera_framebuffer->setGamma(value);
  }
}

void Renderer::setExposureValue(float value) {
  if (camera_framebuffer) {
    camera_framebuffer->setExposure(value);
  }
}

void Renderer::setNoPostProcess() {
  if (camera_framebuffer) {
    camera_framebuffer->setPostProcessDefault();
  }
}

void Renderer::setPostProcessEdge() {
  if (camera_framebuffer) {
    camera_framebuffer->setPostProcessEdge();
  }
}

void Renderer::setPostProcessSharpen() {
  if (camera_framebuffer) {
    camera_framebuffer->setPostProcessSharpen();
  }
}

void Renderer::setPostProcessBlurr() {
  if (camera_framebuffer) {
    camera_framebuffer->setPostProcessBlurr();
  }
}

void Renderer::resetSceneCamera() {
  if (scene_camera) {
    scene_camera->reset();
  }
}

void Renderer::setRasterizerFill() {
  if (scene) {
    scene->setPolygonFill();
  }
}

void Renderer::setRasterizerPoint() {
  if (scene) {
    scene->setPolygonPoint();
  }
}

void Renderer::setRasterizerWireframe() {
  if (scene) {
    scene->setPolygonWireframe();
  }
}

void Renderer::displayBoundingBoxes(bool display) {
  if (scene) {
    scene->displayBoundingBoxes(display);
  }
}
void Renderer::setViewerWidget(GLViewer *widget) {
  gl_widget = widget;
  if (render_pipeline)
    render_pipeline->setContextSwitcher(gl_widget);
}
