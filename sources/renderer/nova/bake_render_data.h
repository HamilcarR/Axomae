#ifndef BAKERENDERDATA_H
#define BAKERENDERDATA_H
#include "Image.h"
#include <QWidget>
#include <internal/common/math/utils_3D.h>
#include <nova/api_engine.h>

class Mesh;
class Drawable;

/**
 *@brief structures related to the management of offline renderers resources
 */
namespace nova_baker_utils {

  struct texture_buffers_t {
    axstd::span<nova::texturing::ImageTexture<uint32_t>> image_alloc_buffer;
  };

  struct primitive_buffers_t {
    axstd::span<nova::primitive::NovaGeoPrimitive> geo_primitive_alloc_buffer;
  };

  struct render_scene_context {
    std::unique_ptr<nova::HdrBufferStruct> buffers;
    int width, height;
    std::unique_ptr<NovaRenderEngineInterface> engine_instance;
    nova::NovaResourceManager *nova_resource_manager{nullptr};
    std::unique_ptr<nova::NovaExceptionManager> nova_exception_manager;
    threading::ThreadPool *thread_pool{nullptr};
    nova::device_shared_caches_t shared_caches;
    nova::gputils::gpu_util_structures_t gpu_util_structures;
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

  struct envmap_memory_s {
    GLuint equirect_glID;
    float *raw_data;
    int width;
    int height;
    int channels;
  };

  struct envmap_data_s {
    std::vector<envmap_memory_s> env_textures;
    unsigned current_envmap_id;
  };

  struct scene_transform_data {
    glm::mat4 root_transformation;
    glm::mat4 root_translation;
    glm::mat4 root_rotation;
  };

  struct engine_data {
    envmap_data_s environment_maps;
    camera_data camera;
    scene_transform_data scene;
    const std::vector<Mesh *> *mesh_list;
    int samples_max, samples_increment, aa_samples;
    int depth_max;
    int num_tiles_w, num_tiles_h;
    int engine_type_flag;
    bool flip_v, use_gpu{false};
    std::string threadpool_tag;
  };

  struct bake_temp_buffers {
    image::ThumbnailImageHolder<float> image_holder;
  };

  struct NovaBakingStructure {
    bake_temp_buffers bake_buffers;
    std::unique_ptr<QWidget> spawned_window;

    void reinitialize() {
      /* Remove reference to the widget. */
      spawned_window = nullptr;
    }
  };

  /* We use this because of the sequence of scene initialization through the renderers :
   * 1) Retrieve original transformation matrices of each mesh.
   * 2) Drawables are built first by building the scene in each renderer :
   * this will introduce the camera + skybox nodes on top of the dependency tree , modifying each mesh transformation
   * 3) Hence we need the original transformations to reconstruct the scene in Nova as we need the unchanged node transformations.
   *
   * Access to drawables gives access to the different VBOs used which we can register using cuda-GL interop.
   * This is required to reduce the memory footprint of a scene by loading only one set of geometry buffers , and just referencing it.
   */
  struct drawable_original_transform {
    Drawable *mesh;
    glm::mat4 mesh_original_transformation;
  };
}  // namespace nova_baker_utils

#endif  // BAKERENDERDATA_H
