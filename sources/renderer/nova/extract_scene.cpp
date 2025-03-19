#include "Drawable.h"
#include "Mesh.h"
#include "aggregate/acceleration_interface.h"
#include "bake.h"
#include "manager/NovaResourceManager.h"
#include "material/nova_material.h"
#include "primitive/PrimitiveInterface.h"
#include "shape/nova_shape.h"
#include <internal/common/axstd/span.h>

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

  primitive_buffers_t allocate_primitive_triangle_buffers(core::memory::ByteArena &memory_pool, std::size_t primitive_number) {
    auto *primitive_buffer = memory_pool.construct<nova::primitive::NovaGeoPrimitive>(primitive_number, false, "Primitive buffer");
    primitive_buffers_t geometry_buffers{};
    geometry_buffers.geo_primitive_alloc_buffer = axstd::span<nova::primitive::NovaGeoPrimitive>(primitive_buffer, primitive_number);
    return geometry_buffers;
  }

  /**
   * Each mesh has PBR_PIPELINE_TEX_NUM number of textures
   */
  template<class T>
  T *allocate_type_texture_buffer(core::memory::ByteArena &memory_arena,
                                  std::size_t num_textures,
                                  std::size_t alignment = core::memory::PLATFORM_ALIGN) {
    return memory_arena.allocate(sizeof(T) * num_textures, "", alignment);
  }

  static std::vector<Mesh *> retrieve_meshes_from_drawables(const std::vector<drawable_original_transform> &drawables) {
    std::vector<Mesh *> meshes;
    meshes.reserve(drawables.size());
    for (const auto &elem : drawables)
      meshes.push_back(elem.mesh->getMeshPointer());
    return meshes;
  }

  struct resource_holders_inits_s {
    std::size_t triangle_mesh_number;
    std::size_t triangle_number;
    std::size_t dielectrics_number;
    std::size_t diffuse_number;
    std::size_t conductors_number;
    std::size_t image_texture_number;
    std::size_t constant_texture_number;
  };

  static void initialize_resources_holders(nova::NovaResourceManager &manager, const resource_holders_inits_s &resrc) {
    auto &shape_reshdr = manager.getShapeData();
    nova::shape::shape_init_record_s shape_init_data{};
    shape_init_data.total_triangle_meshes = resrc.triangle_mesh_number;
    shape_init_data.total_triangles = resrc.triangle_number;
    shape_reshdr.init(shape_init_data);

    nova::material::material_init_record_s material_init_data{};
    material_init_data.conductors_size = resrc.conductors_number;
    material_init_data.dielectrics_size = resrc.dielectrics_number;
    material_init_data.diffuse_size = resrc.diffuse_number;
    auto &material_resrc = manager.getMaterialData();
    material_resrc.init(material_init_data);

    nova::texturing::texture_init_record_s texture_init_data{};
    texture_init_data.constant_texture_size = resrc.constant_texture_number;
    texture_init_data.image_texture_size = resrc.image_texture_number;
    auto &texture_resrc = manager.getTexturesData();
    texture_resrc.init(texture_init_data);
  }

  void build_scene(const std::vector<drawable_original_transform> &drawables_orig_transfo, nova::NovaResourceManager &manager) {
    core::memory::ByteArena &memory_pool = manager.getMemoryPool();
    std::vector<Mesh *> meshes = retrieve_meshes_from_drawables(drawables_orig_transfo);
    /* Allocate for triangles */
    std::size_t primitive_number = compute_primitive_number(meshes);
    primitive_buffers_t primitive_buffers = allocate_primitive_triangle_buffers(memory_pool, primitive_number);

    resource_holders_inits_s resrc{};
    resrc.triangle_mesh_number = meshes.size();
    resrc.triangle_number = primitive_number;
    resrc.conductors_number = meshes.size();
    resrc.diffuse_number = meshes.size();
    resrc.dielectrics_number = meshes.size();
    resrc.image_texture_number = meshes.size() * PBR_PIPELINE_TEX_NUM;
    initialize_resources_holders(manager, resrc);

    std::size_t alloc_offset_primitives = 0, mesh_index = 0;
    for (const drawable_original_transform &dtf : drawables_orig_transfo) {
      nova::material::NovaMaterialInterface material = setup_material_data(*dtf.mesh, manager);
      setup_geometry_data(primitive_buffers, dtf, alloc_offset_primitives, material, manager, mesh_index);
      mesh_index++;
    }
  }

  nova::aggregate::DefaultAccelerator build_api_managed_acceleration_structure(nova::aggregate::primitive_aggregate_data_s primitive_geometry) {
    nova::aggregate::DefaultAccelerator acceleration;
    acceleration.build(primitive_geometry);
    return acceleration;
  }

}  // namespace nova_baker_utils
