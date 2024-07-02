#include "Camera.h"
#include "CameraInterface.h"
#include "DrawEngine.h"
#include "GenericException.h"
#include "Image.h"
#include "ThreadPool.h"
#include "bake.h"
#include "manager/NovaResourceManager.h"
namespace exception {
  class NullMeshListException : public GenericException {
   public:
    NullMeshListException() : GenericException() { saveErrorString("Error reading scene description."); }
  };
}  // namespace exception
namespace nova_baker_utils {
  void initialize_matrices(const camera_data &scene_camera,
                           const scene_transform_data &scene_data,
                           nova::scene::SceneTransformations &nova_scene_transformations,
                           nova::camera::CameraResourcesHolder &nova_camera_structure) {

    /* Camera data*/
    nova_camera_structure.setUpVector(scene_camera.up_vector);
    nova_camera_structure.setProjection(scene_camera.projection);
    nova_camera_structure.setInvProjection(glm::inverse(scene_camera.projection));
    nova_camera_structure.setView(scene_camera.view);
    nova_camera_structure.setInvView(glm::inverse(nova_camera_structure.getView()));
    nova_camera_structure.setPosition(scene_camera.position);
    nova_camera_structure.setDirection(scene_camera.direction);

    /* Scene root transformations */
    nova_scene_transformations.setTranslation(scene_data.root_translation);
    nova_scene_transformations.setInvTranslation(glm::inverse(scene_data.root_translation));
    nova_scene_transformations.setRotation(scene_data.root_rotation);
    nova_scene_transformations.setInvRotation(glm::inverse(scene_data.root_rotation));
    nova_scene_transformations.setModel(scene_data.root_transformation);
    nova_scene_transformations.setInvModel(glm::inverse(scene_data.root_transformation));
    nova_scene_transformations.setPvm(scene_camera.projection * scene_camera.view * scene_data.root_transformation);
    nova_scene_transformations.setInvPvm(glm::inverse(nova_scene_transformations.getPvm()));
    nova_scene_transformations.setVm(nova_camera_structure.getView() * nova_scene_transformations.getModel());
    nova_scene_transformations.setInvVm(glm::inverse(nova_scene_transformations.getVm()));
    nova_scene_transformations.setNormalMatrix(glm::mat3(glm::transpose(nova_scene_transformations.getInvModel())));
  }

  void initialize_engine_opts(const engine_data &engine_opts, nova::engine::EngineResourcesHolder &engine_resources_holder) {
    engine_resources_holder.setAliasingSamples(engine_opts.aa_samples);
    engine_resources_holder.setCancelPtr(engine_opts.stop_render_ptr);
    engine_resources_holder.setMaxDepth(engine_opts.depth_max);
    engine_resources_holder.setMaxSamples(engine_opts.samples_max);
    engine_resources_holder.setSampleIncrement(engine_opts.samples_increment);
    engine_resources_holder.setTilesHeight(engine_opts.num_tiles_w);
    engine_resources_holder.setTilesWidth(engine_opts.num_tiles_h);
    engine_resources_holder.setVAxisInversed(engine_opts.flip_v);
  }

  void initialize_environment_texture(const scene_envmap &envmap, nova::texturing::TextureRawData &texture_raw_data) {
    texture_raw_data.raw_data = &envmap.hdr_envmap->data;
    texture_raw_data.channels = envmap.hdr_envmap->metadata.channels;
    texture_raw_data.width = envmap.hdr_envmap->metadata.width;
    texture_raw_data.height = envmap.hdr_envmap->metadata.height;
  }

  void initialize_manager(const engine_data &engine_opts, nova::NovaResourceManager &manager) {
    /* Initialize every matrix of the scene , and camera structures*/
    nova::camera::CameraResourcesHolder &camera_resources_holder = manager.getCameraData();
    nova::scene::SceneTransformations &scene_transformations = manager.getSceneTransformation();
    initialize_matrices(engine_opts.camera, engine_opts.scene, scene_transformations, camera_resources_holder);

    /* Initialize engine options */
    nova::engine::EngineResourcesHolder &engine_resources_holder = manager.getEngineData();
    initialize_engine_opts(engine_opts, engine_resources_holder);

    /* Environment map */
    initialize_environment_texture(engine_opts.envmap, manager.getEnvmapData());

    /* Build Scene */
    if (!engine_opts.mesh_list)
      throw exception::NullMeshListException();
    build_scene(*engine_opts.mesh_list, manager);
    build_acceleration_structure(manager);
  }

  void bake_scene(render_scene_data &rendering_data) {
    nova::draw(rendering_data.buffers.get(),
               rendering_data.width,
               rendering_data.height,
               rendering_data.engine_instance.get(),
               rendering_data.thread_pool,
               rendering_data.nova_resource_manager.get()

    );
  }

  void cancel_render(engine_data &data) { *data.stop_render_ptr = true; }

  std::unique_ptr<NovaRenderEngineInterface> create_engine(const engine_data &engine_type) {
    if (engine_type.engine_type_flag & RGB) {
      return std::make_unique<nova::NovaRenderEngineLR>();
    }
    AX_UNREACHABLE
    return nullptr;
  }

  void synchronize_render_threads(render_scene_data &scene_data) {
    if (scene_data.thread_pool) {
      scene_data.thread_pool->fence();
    }
  }

}  // namespace nova_baker_utils