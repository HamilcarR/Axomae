#include "Drawable.h"
#include "EnvmapTextureManager.h"
#include "Mesh.h"
#include "PackedGLGeometryBuffer.h"
#include "api_material.h"
#include "bake.h"
#include "extract_scene_internal.h"
#include "nova/bake_render_data.h"
#include <internal/common/axstd/span.h>
#include <nova/api_engine.h>

namespace nova_baker_utils {

  static std::size_t compute_primitive_number(const std::vector<Mesh *> &meshes, int indices_padding = 3) {
    std::size_t acc = 0;
    for (const auto &elem : meshes) {
      const Object3D &geometry = elem->getGeometry();
      for (int i = 0; i < geometry.indices.size(); i += indices_padding)
        acc++;
    }
    return acc;
  }

  /**
   * Each mesh has PBR_PIPELINE_TEX_NUM number of textures
   */
  template<class T>
  T *allocate_type_texture_buffer(axstd::ByteArena &memory_arena, std::size_t num_textures, std::size_t alignment = axstd::PLATFORM_ALIGN) {
    return memory_arena.allocate(sizeof(T) * num_textures, "", alignment);
  }

  static std::vector<Mesh *> retrieve_meshes_from_drawables(const std::vector<drawable_original_transform> &drawables) {
    std::vector<Mesh *> meshes;
    meshes.reserve(drawables.size());
    for (const auto &elem : drawables)
      meshes.push_back(elem.mesh->getMeshPointer());
    return meshes;
  }

  void setup_envmaps(const EnvmapTextureManager &envmap_manager, nova_baker_utils::envmap_data_s &envmap_data) {
    axstd::span<const texture::envmap::EnvmapTextureGroup> baked_envmaps = envmap_manager.getBakesViews();
    nova_baker_utils::envmap_data_s envmap_infos;
    std::vector<nova_baker_utils::envmap_memory_s> &envmap_data_collection = envmap_infos.env_textures;
    unsigned current_envmap_id = 0;
    auto gl_equi2D_id = envmap_manager.getCurrentEnvmapGroup().equirect_gl_id;
    for (const auto &envmap : baked_envmaps) {
      nova_baker_utils::envmap_memory_s data{};
      data.equirect_glID = envmap.equirect_gl_id;
      if (data.equirect_glID == gl_equi2D_id)
        envmap_infos.current_envmap_id = current_envmap_id;
      AX_ASSERT_NOTNULL(envmap.metadata);
      data.width = envmap.metadata->metadata.width;
      data.height = envmap.metadata->metadata.height;
      data.raw_data = envmap.metadata->data().data();
      data.channels = envmap.metadata->metadata.channels;
      envmap_data_collection.push_back(data);
      current_envmap_id++;
    }
    envmap_data = envmap_infos;
  }

  void build_scene(const std::vector<drawable_original_transform> &drawables_orig_transfo, nova::Scene *scene) {
    std::vector<Mesh *> meshes = retrieve_meshes_from_drawables(drawables_orig_transfo);
    for (const drawable_original_transform &dtf : drawables_orig_transfo) {
      nova::TrimeshPtr mesh = nova::create_trimesh();
      nova::MaterialPtr material = nova::create_material();
      setup_mesh(dtf, *mesh);
      setup_material(dtf, *material);
      scene->addMesh(std::move(mesh), std::move(material));
    }
  }

  nova::aggregate::DefaultAccelerator build_api_managed_acceleration_structure(nova::aggregate::primitive_aggregate_data_s primitive_geometry) {
    nova::aggregate::DefaultAccelerator acceleration;
    acceleration.build(primitive_geometry);
    return acceleration;
  }

  std::unique_ptr<nova::aggregate::DeviceAcceleratorInterface> build_device_managed_acceleration_structure(
      nova::aggregate::primitive_aggregate_data_s primitive_geometry) {
    auto device_builder = nova::aggregate::DeviceAcceleratorInterface::make();
    device_builder->build(primitive_geometry);
    return device_builder;
  };

}  // namespace nova_baker_utils
