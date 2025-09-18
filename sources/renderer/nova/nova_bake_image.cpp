#include "Camera.h"
#include "Image.h"
#include "api_common.h"
#include "api_renderoptions.h"
#include "bake.h"
#include "internal/debug/Logger.h"
#include "internal/macro/project_macros.h"
#include "nova/bake_render_data.h"
#include <internal/common/exception/GenericException.h>
#include <internal/thread/worker/ThreadPool.h>
#include <nova/api_engine.h>

namespace exception {
  class NullMeshListException : public GenericException {
   public:
    NullMeshListException() : GenericException() { saveErrorString("Error reading scene description."); }
  };
}  // namespace exception
namespace nova_baker_utils {

  // TODO : DELETE after nova realtime rt refactor.
  void initialize_scene_data(const camera_data &scene_camera,
                             const scene_transform_data &scene_data,
                             nova::scene::SceneTransformations &nova_scene_transformations,
                             nova::camera::CameraResourcesHolder &nova_camera_structure) {

    /* Camera data */
    glm::vec3 up_vector = scene_camera.up_vector;
    glm::mat4 P = scene_camera.projection;
    glm::mat4 inv_P = glm::inverse(P);
    // this syncs the view from nova to the Realtime renderer.
    glm::mat4 V = glm::rotate(scene_camera.view, glm::radians(90.f), glm::vec3(-1.f, 0.f, 0.f));
    glm::mat4 inv_V = glm::inverse(V);
    glm::vec3 position = scene_camera.position;
    glm::vec3 direction = scene_camera.direction;
    float far_clip = scene_camera.far;
    float near_clip = scene_camera.near;
    int screen_width = scene_camera.width;
    int screen_height = scene_camera.height;
    float fov = scene_camera.fov;

    nova_camera_structure.up_vector = up_vector;
    nova_camera_structure.P = P;
    nova_camera_structure.inv_P = inv_P;
    nova_camera_structure.V = V;
    nova_camera_structure.inv_V = inv_V;
    nova_camera_structure.position = position;
    nova_camera_structure.direction = direction;
    nova_camera_structure.far = far_clip;
    nova_camera_structure.near = near_clip;
    nova_camera_structure.screen_width = screen_width;
    nova_camera_structure.screen_height = screen_height;
    nova_camera_structure.fov = fov;
  }

  // TODO: delete
  void initialize_engine_opts(const engine_data &engine_opts, nova::engine::EngineResourcesHolder &engine_resources_holder) {
    engine_resources_holder.aliasing_samples = engine_opts.aa_samples;
    engine_resources_holder.is_rendering = true;
    engine_resources_holder.max_depth = engine_opts.depth_max;
    engine_resources_holder.renderer_max_samples = engine_opts.samples_max;
    engine_resources_holder.sample_increment = engine_opts.samples_increment;
    engine_resources_holder.tiles_width = engine_opts.num_tiles_w;
    engine_resources_holder.tiles_height = engine_opts.num_tiles_h;
    engine_resources_holder.vertical_invert = engine_opts.flip_v;
    engine_resources_holder.threadpool_tag = engine_opts.threadpool_tag;
    engine_resources_holder.integrator_flag = engine_opts.engine_type_flag;
  }

  // TODO delete
  void initialize_environment_maps(const envmap_data_s &envmaps, nova::texturing::TextureResourcesHolder &texture_resources_holder) {
    texture_resources_holder.allocateEnvironmentMaps(envmaps.env_textures.size());
    texture_resources_holder.setEnvmapId(envmaps.current_envmap_id);
    for (const auto &element : envmaps.env_textures) {
      std::size_t index = texture_resources_holder.addTexture(
          element.raw_data, element.width, element.height, element.channels, true, false, element.equirect_glID);
      texture_resources_holder.addNovaTexture<nova::texturing::EnvmapTexture>(index);
    }
  }

  void initialize_envmaps(const envmap_data_s &envmaps, nova::Scene &nova_scene) {
    nova_scene.clearEnvmaps();
    for (const auto &element : envmaps.env_textures) {
      nova::TexturePtr texture = nova::create_texture();
      texture->setTextureBuffer(element.raw_data);
      texture->setChannels(element.channels);
      texture->setHeight(element.height);
      texture->setWidth(element.width);
      texture->setInteropID(element.equirect_glID);
      texture->invertX();
      nova_scene.addEnvmap(std::move(texture));
    }

    if (nova_scene.useEnvmap(envmaps.current_envmap_id) != nova::SUCCESS)
      LOG("Unable to set envmap ID.", LogLevel::CRITICAL);
  }

  nova::CameraPtr initialize_camera(const camera_data &scene_camera) {
    /* Camera data */
    glm::vec3 up_vector = scene_camera.up_vector;
    glm::mat4 P = scene_camera.projection;
    // this syncs the view from nova to the Realtime renderer.
    glm::mat4 V = glm::rotate(scene_camera.view, glm::radians(90.f), glm::vec3(-1.f, 0.f, 0.f));
    glm::vec3 position = scene_camera.position;
    glm::vec3 direction = scene_camera.direction;
    float far_clip = scene_camera.far;
    float near_clip = scene_camera.near;
    int screen_width = scene_camera.width;
    int screen_height = scene_camera.height;
    float fov = scene_camera.fov;

    nova::CameraPtr camera = nova::create_camera();
    AX_ASSERT_NOTNULL(camera);

    camera->setUpVector(glm::value_ptr(up_vector));
    camera->setMatrices(glm::value_ptr(P), glm::value_ptr(V));
    camera->setPosition(glm::value_ptr(position));
    camera->setDirection(glm::value_ptr(direction));
    camera->setClipPlaneFar(far_clip);
    camera->setClipPlaneNear(near_clip);
    camera->setResolutionWidth(screen_width);
    camera->setResolutionHeight(screen_height);
    camera->setFov(fov);
    return camera;
  }

  nova::RenderOptionsPtr initialize_options(const engine_data &engine_opts) {
    nova::RenderOptionsPtr render_options = nova::create_renderoptions();
    AX_ASSERT_NOTNULL(render_options);

    render_options->setAliasingSamples(engine_opts.aa_samples);
    render_options->setMaxDepth(engine_opts.depth_max);
    render_options->setMaxSamples(engine_opts.samples_max);
    render_options->setSamplesIncrement(engine_opts.samples_increment);
    render_options->setTileDimension(engine_opts.num_tiles_w, engine_opts.num_tiles_h);
    if (engine_opts.flip_v)
      render_options->flipV();
    render_options->setIntegratorFlag(engine_opts.engine_type_flag);
    return render_options;
  }

  void initialize_engine(const engine_data &engine_opts, nova::Engine &nova_engine) {

    nova::Scene *nova_scene = nova_engine.getScene();
    if (!nova_scene) {
      LOG("Invalid Nova Scene state.", LogLevel::CRITICAL);
      return;
    }
    /* Initialize main camera.*/
    nova::CameraPtr camera = initialize_camera(engine_opts.camera);

    /* Initialize engine options.*/
    nova::RenderOptionsPtr options = initialize_options(engine_opts);

    /* Initialize environment maps.*/
    initialize_envmaps(engine_opts.environment_maps, *nova_scene);

    nova_engine.setRenderOptions(std::move(options));
    nova_scene->clearCameras();
    nova_scene->addCamera(std::move(camera));
    nova_scene->useCamera(0);
  }

  void bake_scene(render_scene_context &rendering_data) {
    nova::nova_eng_internals interns{rendering_data.nova_resource_manager, rendering_data.nova_exception_manager.get()};
    nova::draw(rendering_data.buffers.get(),
               rendering_data.width,
               rendering_data.height,
               rendering_data.engine_instance.get(),
               rendering_data.thread_pool,
               interns

    );
  }

  void cancel_render(render_scene_context &rendering_data) { rendering_data.nova_resource_manager->getEngineData().is_rendering = false; }
  void start_render(render_scene_context &rendering_data) { rendering_data.nova_resource_manager->getEngineData().is_rendering = true; }

  std::unique_ptr<NovaRenderEngineInterface> create_engine(const engine_data &engine_type) { return std::make_unique<nova::NovaRenderEngineLR>(); }

  void synchronize_render_threads(render_scene_context &scene_data, const std::string &tag) {
    if (scene_data.thread_pool) {
      scene_data.thread_pool->fence(tag);
    }
  }
}  // namespace nova_baker_utils
