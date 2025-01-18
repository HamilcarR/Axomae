#ifndef BAKERENDERDATA_H
#define BAKERENDERDATA_H
#include "DrawEngine.h"
#include "Image.h"
#include "manager/NovaResourceManager.h"
#include <QWidget>

class Mesh;
class Drawable;

/**
 *@brief structures related to the management of offline renderers resources
 */
namespace nova_baker_utils {

  struct material_buffers_t {
    axstd::span<nova::material::NovaConductorMaterial> conductor_alloc_buffer;
    axstd::span<nova::material::NovaDielectricMaterial> dielectric_alloc_buffer;
    axstd::span<nova::material::NovaDiffuseMaterial> diffuse_alloc_buffer;
  };

  struct texture_buffers_t {
    axstd::span<nova::texturing::ImageTexture> image_alloc_buffer;
  };

  struct primitive_buffers_t {
    axstd::span<nova::shape::Triangle> triangle_alloc_buffer;
    axstd::span<nova::primitive::NovaGeoPrimitive> geo_primitive_alloc_buffer;
  };

  struct bake_buffers_storage_t {
    material_buffers_t material_buffers;
    texture_buffers_t texture_buffers;
    primitive_buffers_t primitive_buffers;
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

  struct bake_temp_buffers {
    image::ThumbnailImageHolder<float> image_holder;
    std::vector<float> accumulator;
    std::vector<float> partial;
    std::vector<float> depth;
  };

  struct NovaBakingStructure {
    bake_temp_buffers bake_buffers;
    std::unique_ptr<QWidget> spawned_window;
    render_scene_context render_context;
    std::thread worker_baking_thread;

    void reinitialize() {
      auto &nova_resource_manager = render_context.nova_resource_manager;
      if (nova_resource_manager)
        nova_resource_manager->getEngineData().is_rendering = false;
      if (worker_baking_thread.joinable())
        worker_baking_thread.join();

      /* Clear the render buffers. */
      bake_buffers.image_holder.clear();
      bake_buffers.accumulator.clear();
      bake_buffers.partial.clear();
      /* Remove reference to the widget. */
      spawned_window = nullptr;
      /* If gpu is used , cleanup whatever has been allocated */
      nova::gputils::cleanup_gpu_structures(render_context.gpu_util_structures);
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
