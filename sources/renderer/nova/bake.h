#ifndef NOVABAKE_H
#define NOVABAKE_H
#include "bake_render_data.h"
#include <internal/common/axstd/span.h>
#include <nova/NovaAPI.h>

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

  void build_scene(const std::vector<drawable_original_transform> &drawables, nova::NovaResourceManager &manager);
  nova::aggregate::DefaultAccelerator build_api_managed_acceleration_structure(nova::aggregate::primitive_aggregate_data_s primitive_geometry);
  std::unique_ptr<nova::aggregate::DeviceAcceleratorInterface> build_device_managed_acceleration_structure(
      nova::aggregate::primitive_aggregate_data_s primitive_geometry);

  /* Takes an initialized NovaResourceManager.*/
  void bake_scene(render_scene_context &rendering_data);
  void bake_scene_gpu(render_scene_context &rendering_data, nova::gputils::gpu_util_structures_t &gpu_structures);
  void initialize_nova_manager(const engine_data &render_data, nova::NovaResourceManager &nova_resource_manager);
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
  uint64_t get_error_status(const nova::NovaExceptionManager &exception_manager);
  std::vector<nova::exception::ERROR> get_error_list(const nova::NovaExceptionManager &exception_manager);

}  // namespace nova_baker_utils
#endif  // BAKE_H
