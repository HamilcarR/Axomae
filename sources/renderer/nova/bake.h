#ifndef NOVABAKE_H
#define NOVABAKE_H
#include "NovaInterface.h"
#include "Texture.h"

#include <engine/nova_exception.h>

class Camera;
class TextureGroup;
class Mesh;

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
  const nova::material::NovaMaterialInterface *extract_materials(const Mesh *mesh, nova::NovaResourceManager &manager);
  void build_scene(const std::vector<Mesh *> &meshes, nova::NovaResourceManager &manager);
  /* Uses the primitives in the SceneResourceHolder to build the acceleration structure stored in the manager's Accelerator structure*/
  void build_acceleration_structure(nova::NovaResourceManager &manager);
  void transform_vertices(const geometry::face_data_tri &tri_primitive, const glm::mat4 &final_transfo, glm::vec3 vertices[3]);
  void transform_normals(const geometry::face_data_tri &tri_primitive, const glm::mat3 &normal_matrix, glm::vec3 normals[3]);
  void transform_tangents(const geometry::face_data_tri &tri_primitive, const glm::mat3 &normal_matrix, glm::vec3 tangents[3]);
  void transform_bitangents(const geometry::face_data_tri &tri_primitive, const glm::mat3 &normal_matrix, glm::vec3 bitangents[3]);
  void extract_uvs(const geometry::face_data_tri &tri_primitive, glm::vec2 textures[3]);

  struct render_scene_data {
    std::unique_ptr<nova::HdrBufferStruct> buffers;
    int width, height;
    std::unique_ptr<NovaRenderEngineInterface> engine_instance;
    std::unique_ptr<nova::NovaResourceManager> nova_resource_manager;
    threading::ThreadPool *thread_pool;
  };

  struct camera_data {
    glm::mat4 view;
    glm::mat4 projection;
    glm::vec3 up_vector;
    glm::vec3 position;
    glm::vec3 direction;
    int width, height;
    float far, near, fov;
  };

  struct scene_transform_data {
    glm::mat4 root_transformation;
    glm::mat4 root_translation;
    glm::mat4 root_rotation;
  };

  struct scene_envmap {
    image::ImageHolder<float> *hdr_envmap;
  };

  struct engine_data {
    camera_data camera;
    scene_transform_data scene;
    scene_envmap envmap;
    const std::vector<Mesh *> *mesh_list;
    int samples_max, samples_increment, aa_samples;
    int depth_max;
    int num_tiles_w, num_tiles_h;
    int engine_type_flag;
    bool flip_v;
    std::string threadpool_tag;
  };

  /* Takes an initialized NovaResourceManager.*/
  void bake_scene(render_scene_data &rendering_data);
  void bake_scene_gpu(render_scene_data &rendering_data);
  void initialize_manager(const engine_data &render_data, nova::NovaResourceManager &nova_resource_manager);
  void initialize_scene_data(const camera_data &scene_camera,
                             const scene_transform_data &scene_data,
                             nova::scene::SceneTransformations &scene_transform,
                             nova::camera::CameraResourcesHolder &camera_resources_holder);
  void initialize_engine_opts(const engine_data &engine_opts, nova::engine::EngineResourcesHolder &engine_resources_holder);
  void initialize_environment_texture(const scene_envmap &envmap, nova::texturing::TextureRawData &texture_raw_data);
  void cancel_render(engine_data &data);
  std::unique_ptr<NovaRenderEngineInterface> create_engine(const engine_data &engine_type);
  void synchronize_render_threads(render_scene_data &scene_data, const std::string &tag);
  uint64_t get_error_status(const nova::NovaResourceManager &nova_resource_manager);
  std::vector<nova::exception::ERROR> get_error_list(const nova::NovaResourceManager &nova_resource_manager);

}  // namespace nova_baker_utils
#endif  // BAKE_H
