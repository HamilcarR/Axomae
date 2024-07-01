#ifndef NOVABAKE_H
#define NOVABAKE_H
#include "NovaInterface.h"
#include "Texture.h"
#include "rendering/nova_engine.h"

class TextureGroup;
class Mesh;
namespace nova::texturing {
  class ImageTexture;
}
namespace nova::material {
  class NovaMaterialInterface;
}
namespace nova {
  class NovaResourceManager;
}
namespace geometry {
  struct face_data_tri;
}
namespace threading {
  class ThreadPool;
}
const nova::texturing::ImageTexture *extract_texture(const TextureGroup &tgroup, nova::NovaResourceManager &manager, Texture::TYPE type);
const nova::material::NovaMaterialInterface *extract_materials(const Mesh *mesh, nova::NovaResourceManager &manager);
void build_scene(const std::vector<Mesh *> &meshes, nova::NovaResourceManager &manager);
/* Uses the primitives in the SceneResourceHolder to build the acceleration structure stored in the manager's Accelerator structure*/
void build_acceleration_structure(nova::NovaResourceManager &manager);
void transform_vertices(const geometry::face_data_tri &tri_primitive, const glm::mat4 &final_transfo, glm::vec3 vertices[3]);
void transform_normals(const geometry::face_data_tri &tri_primitive, const glm::mat3 &normal_matrix, glm::vec3 normals[3]);
void transform_tangents(const geometry::face_data_tri &tri_primitive, const glm::mat3 &normal_matrix, glm::vec3 tangents[3]);
void transform_bitangents(const geometry::face_data_tri &tri_primitive, const glm::mat3 &normal_matrix, glm::vec3 bitangents[3]);
void extract_uvs(const geometry::face_data_tri &tri_primitive, glm::vec2 textures[3]);

struct nova_bake_struct {
  nova::HdrBufferStruct *buffers;
  int width, height;
  NovaRenderEngineInterface *engine_instance;
  nova::NovaResourceManager *nova_resource_manager;
  threading::ThreadPool *thread_pool;
};

struct nova_render_data {
  int samples_max;
  int depth_max;
  int num_tiles;
  bool *stop_render_ptr;
  const CameraInterface *camera;
  const std::vector<Mesh *> *mesh_list;
};
/* Takes an initialized NovaResourceManager.*/
void bake_scene(nova_bake_struct &rendering_data);
void bake_initialize_manager(nova_bake_struct &bake_struct, const nova_render_data &render_data);
#endif  // BAKE_H
