#ifndef NOVABAKE_H
#define NOVABAKE_H
#include "bake_render_data.h"
#include <internal/common/axstd/span.h>
#include <nova/api_engine.h>

class Camera;
class TextureGroup;
class Drawable;
class EnvmapTextureManager;

namespace nova {
  class NovaResourceManager;

  namespace material {
    class NovaMaterialInterface;
  }

}  // namespace nova

namespace geometry {
  struct face_data_tri;
}
namespace threading {
  class ThreadPool;
}
namespace image {
  template<class T>
  class ImageHolder;
}
namespace nova_baker_utils {

  void build_scene(const std::vector<drawable_original_transform> &drawables, nova::Scene *nova_scene);

  /* Takes an initialized NovaResourceManager.*/
  void bake_scene(render_scene_context &rendering_data);
  void bake_scene_gpu(render_scene_context &rendering_data, nova::gputils::gpu_util_structures_t &gpu_structures);
  void initialize_engine(const engine_data &render_data, nova::Engine &nova_engine);
  nova::CameraPtr initialize_camera(const camera_data &scene_camera);
  nova::RenderOptionsPtr initialize_options(const engine_data &engine_opts);
  void initialize_envmaps(const envmap_data_s &envmaps, nova::Scene &nova_scene);

  // TODO: replace by intialize_scene
  void initialize_scene_data(const camera_data &scene_camera,
                             const scene_transform_data &scene_data,
                             nova::scene::SceneTransformations &scene_transform,
                             nova::camera::CameraResourcesHolder &camera_resources_holder);
  void initialize_engine_opts(const engine_data &engine_opts, nova::engine::EngineResourcesHolder &engine_resources_holder);
  void initialize_environment_maps(const envmap_data_s &envmaps, nova::texturing::TextureResourcesHolder &texture_resources_holder);
  /* Retrieve all envmaps registered.*/
  void setup_envmaps(const EnvmapTextureManager &envmap_manager, nova_baker_utils::envmap_data_s &envmap_data);
  void cancel_render(engine_data &data);
  std::unique_ptr<NovaRenderEngineInterface> create_engine(const engine_data &engine_type);
  void synchronize_render_threads(render_scene_context &scene_data, const std::string &tag);
}  // namespace nova_baker_utils
#endif  // BAKE_H
