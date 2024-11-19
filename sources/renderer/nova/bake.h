#ifndef NOVABAKE_H
#define NOVABAKE_H
#include "NovaInterface.h"
#include "Texture.h"
#include "bake_render_data.h"
#include "boost/core/span.hpp"
#include "engine/nova_exception.h"

class Camera;
class TextureGroup;

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

  const nova::texturing::ImageTexture *extract_texture(const TextureGroup &tgroup, nova::NovaResourceManager &manager, GenericTexture::TYPE type);
  nova::material::texture_pack extract_materials(texture_buffers_t &texture_buffers,
                                                 std::size_t offset,
                                                 const Mesh *mesh,
                                                 nova::NovaResourceManager &manager);

  void setup_geometry_data(primitive_buffers_t &geometry_buffers,
                           Mesh *mesh,
                           std::size_t &alloc_offset_primitives,
                           nova::material::NovaMaterialInterface &material,
                           nova::NovaResourceManager &manager);
  nova::material::NovaMaterialInterface setup_material_data(material_buffers_t &material_buffers,
                                                            texture_buffers_t &texture_buffers,
                                                            const Mesh *mesh,
                                                            nova::NovaResourceManager &manager,
                                                            std::size_t &alloc_offset_materials);
  bake_buffers_storage_t build_scene(const std::vector<Mesh *> &meshes, nova::NovaResourceManager &manager);
  /* Uses the primitives in the SceneResourceHolder to build the acceleration structure stored in the manager's Accelerator structure*/
  void build_acceleration_structure(nova::NovaResourceManager &manager);
  void transform_vertices(const geometry::face_data_tri &tri_primitive, const glm::mat4 &final_transfo, glm::vec3 vertices[3]);
  void transform_normals(const geometry::face_data_tri &tri_primitive, const glm::mat3 &normal_matrix, glm::vec3 normals[3]);
  void transform_tangents(const geometry::face_data_tri &tri_primitive, const glm::mat3 &normal_matrix, glm::vec3 tangents[3]);
  void transform_bitangents(const geometry::face_data_tri &tri_primitive, const glm::mat3 &normal_matrix, glm::vec3 bitangents[3]);
  void extract_uvs(const geometry::face_data_tri &tri_primitive, glm::vec2 textures[3]);
  primitive_buffers_t allocate_primitive_triangle_buffers(core::memory::ByteArena &memory_pool, std::size_t number_elements);
  material_buffers_t allocate_materials_buffers(core::memory::ByteArena &memory_pool, std::size_t number_elements);

  /* Takes an initialized NovaResourceManager.*/
  void bake_scene(render_scene_context &rendering_data);
  void bake_scene_gpu(render_scene_context &rendering_data);
  void initialize_manager(const engine_data &render_data,
                          nova::NovaResourceManager &nova_resource_manager,
                          nova::device_shared_caches_t &shared_caches);
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
