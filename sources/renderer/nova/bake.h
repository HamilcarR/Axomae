#ifndef NOVABAKE_H
#define NOVABAKE_H
#include "NovaInterface.h"
#include "Texture.h"
#include "bake_render_data.h"
#include "engine/nova_exception.h"
#include <internal/common/axstd/span.h>

class Camera;
class TextureGroup;
class Drawable;

namespace nova {
  class NovaResourceManager;

  namespace material {
    class NovaMaterialInterface;
  }
  namespace texturing {
    class ImageTexture;
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

  void setup_geometry_data(primitive_buffers_t &geometry_buffers,
                           const drawable_original_transform &drawable,
                           std::size_t &alloc_offset_primitives,
                           nova::material::NovaMaterialInterface &material,
                           nova::NovaResourceManager &manager,
                           std::size_t mesh_index);

  nova::material::NovaMaterialInterface setup_material_data(material_buffers_t &material_buffers,
                                                            texture_buffers_t &texture_buffers,
                                                            const Drawable &drawable,
                                                            nova::NovaResourceManager &manager,
                                                            std::size_t &alloc_offset_textures,
                                                            std::size_t &alloc_offset_materials);
  bake_buffers_storage_t build_scene(const std::vector<drawable_original_transform> &drawables, nova::NovaResourceManager &manager);
  nova::aggregate::DefaultAccelerator build_api_managed_acceleration_structure(nova::aggregate::primitive_aggregate_data_s primitive_geometry);
  primitive_buffers_t allocate_primitive_triangle_buffers(core::memory::ByteArena &memory_pool, std::size_t number_elements);
  material_buffers_t allocate_materials_buffers(core::memory::ByteArena &memory_pool, std::size_t number_elements);

  /* Takes an initialized NovaResourceManager.*/
  void bake_scene(render_scene_context &rendering_data);
  void bake_scene_gpu(render_scene_context &rendering_data, nova::gputils::gpu_util_structures_t &gpu_structures);
  void initialize_nova_manager(const engine_data &render_data, nova::NovaResourceManager &nova_resource_manager);
  void initialize_scene_data(const camera_data &scene_camera,
                             const scene_transform_data &scene_data,
                             nova::scene::SceneTransformations &scene_transform,
                             nova::camera::CameraResourcesHolder &camera_resources_holder);
  void initialize_engine_opts(const engine_data &engine_opts, nova::engine::EngineResourcesHolder &engine_resources_holder);
  void initialize_environment_texture(const scene_envmap &envmap, nova::texturing::TextureRawData &texture_raw_data);
  void cancel_render(engine_data &data);
  std::unique_ptr<NovaRenderEngineInterface> create_engine(const engine_data &engine_type);
  void synchronize_render_threads(render_scene_context &scene_data, const std::string &tag);
  uint64_t get_error_status(const nova::NovaExceptionManager &exception_manager);
  std::vector<nova::exception::ERROR> get_error_list(const nova::NovaExceptionManager &exception_manager);

}  // namespace nova_baker_utils
#endif  // BAKE_H
