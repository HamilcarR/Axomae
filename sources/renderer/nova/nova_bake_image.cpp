#include "Camera.h"
#include "DrawEngine.h"
#include "Image.h"
#include "bake.h"
#include "internal/common/exception/GenericException.h"
#include "internal/thread/worker/ThreadPool.h"
#include "manager/NovaResourceManager.h"
#include "nova_gpu_utils.h"

namespace exception {
  class NullMeshListException : public GenericException {
   public:
    NullMeshListException() : GenericException() { saveErrorString("Error reading scene description."); }
  };
}  // namespace exception
namespace nova_baker_utils {
  void initialize_scene_data(const camera_data &scene_camera,
                             const scene_transform_data &scene_data,
                             nova::scene::SceneTransformations &nova_scene_transformations,
                             nova::camera::CameraResourcesHolder &nova_camera_structure) {

    /* Camera data*/
    nova_camera_structure.up_vector = scene_camera.up_vector;
    nova_camera_structure.P = scene_camera.projection;
    nova_camera_structure.inv_P = glm::inverse(scene_camera.projection);
    nova_camera_structure.V = scene_camera.view;
    nova_camera_structure.inv_V = glm::inverse(nova_camera_structure.V);
    nova_camera_structure.position = scene_camera.position;
    nova_camera_structure.direction = scene_camera.direction;
    nova_camera_structure.far = scene_camera.far;
    nova_camera_structure.near = scene_camera.near;
    nova_camera_structure.screen_width = scene_camera.width;
    nova_camera_structure.screen_height = scene_camera.height;
    nova_camera_structure.fov = scene_camera.fov;

    /* Scene root transformations */
    nova_scene_transformations.T = scene_data.root_translation;
    nova_scene_transformations.inv_T = glm::inverse(scene_data.root_translation);
    nova_scene_transformations.R = scene_data.root_rotation;
    nova_scene_transformations.inv_R = glm::inverse(scene_data.root_rotation);
    nova_scene_transformations.M = scene_data.root_transformation;
    nova_scene_transformations.inv_M = glm::inverse(scene_data.root_transformation);
    nova_scene_transformations.PVM = scene_camera.projection * scene_camera.view * scene_data.root_transformation;
    nova_scene_transformations.inv_PVM = glm::inverse(nova_scene_transformations.PVM);
    nova_scene_transformations.VM = nova_camera_structure.V * nova_scene_transformations.M;
    nova_scene_transformations.inv_VM = glm::inverse(nova_scene_transformations.VM);
    nova_scene_transformations.N = glm::mat3(glm::transpose(nova_scene_transformations.inv_M));
  }

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

  void initialize_environment_texture(const scene_envmap &envmap, nova::texturing::TextureResourcesHolder &resrc) {
    float *raw_data = envmap.hdr_envmap->data.data();
    int channels = envmap.hdr_envmap->metadata.channels;
    int width = envmap.hdr_envmap->metadata.width;
    int height = envmap.hdr_envmap->metadata.height;

    resrc.setupEnvmap(raw_data, width, height, channels);
  }

  void initialize_nova_manager(const engine_data &engine_opts, nova::NovaResourceManager &manager) {
    /* Initialize every matrix of the scene , and camera structures*/
    nova::camera::CameraResourcesHolder &camera_resources_holder = manager.getCameraData();
    nova::scene::SceneTransformations &scene_transformations = manager.getSceneTransformation();
    initialize_scene_data(engine_opts.camera, engine_opts.scene, scene_transformations, camera_resources_holder);

    /* Initialize engine options */
    nova::engine::EngineResourcesHolder &engine_resources_holder = manager.getEngineData();
    initialize_engine_opts(engine_opts, engine_resources_holder);

    /* Environment map */
    initialize_environment_texture(engine_opts.envmap, manager.getTexturesData());
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
  void bake_scene_gpu(render_scene_context &rendering_data, nova::gputils::gpu_util_structures_t &gpu_structures) {
    nova::nova_eng_internals interns{rendering_data.nova_resource_manager, rendering_data.nova_exception_manager.get()};
    gpu_draw(rendering_data.buffers.get(), rendering_data.width, rendering_data.height, interns, gpu_structures);
  }
  void cancel_render(render_scene_context &rendering_data) { rendering_data.nova_resource_manager->getEngineData().is_rendering = false; }
  void start_render(render_scene_context &rendering_data) { rendering_data.nova_resource_manager->getEngineData().is_rendering = true; }

  std::unique_ptr<NovaRenderEngineInterface> create_engine(const engine_data &engine_type) { return std::make_unique<nova::NovaRenderEngineLR>(); }

  void synchronize_render_threads(render_scene_context &scene_data, const std::string &tag) {
    if (scene_data.thread_pool) {
      scene_data.thread_pool->fence(tag);
    }
  }
  uint64_t get_error_status(const nova::NovaExceptionManager &exception_manager) { return exception_manager.checkErrorStatus(); }
  std::vector<nova::exception::ERROR> get_error_list(const nova::NovaExceptionManager &exception_manager) { return exception_manager.getErrorList(); }
}  // namespace nova_baker_utils