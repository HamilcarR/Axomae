#include "Drawable.h"
#include "MaterialInterface.h"
#include "Mesh.h"
#include "TextureGroup.h"
#include "aggregate/nova_acceleration.h"
#include "bake.h"
#include "manager/NovaResourceManager.h"
#include "material/nova_material.h"
#include "primitive/nova_primitive.h"
#include "shape/nova_shape.h"
#include <internal/common/axstd/span.h>
#include <internal/macro/project_macros.h>

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
    auto *triangle_buffer = memory_pool.construct<nova::shape::Triangle>(primitive_number, false, "Triangle buffer");
    auto *primitive_buffer = memory_pool.construct<nova::primitive::NovaGeoPrimitive>(primitive_number, false, "Primitive buffer");
    primitive_buffers_t geometry_buffers{};
    geometry_buffers.geo_primitive_alloc_buffer = axstd::span<nova::primitive::NovaGeoPrimitive>(primitive_buffer, primitive_number);
    geometry_buffers.triangle_alloc_buffer = axstd::span<nova::shape::Triangle>(triangle_buffer, primitive_number);
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

  static void initialize_resources_holders(nova::NovaResourceManager &manager, std::size_t meshes_number) {
    auto &shape_reshdr = manager.getShapeData();
    nova::shape::shape_init_record_t init_data{};
    init_data.total_triangle_meshes = meshes_number;
    shape_reshdr.init(init_data);
  }

  bake_buffers_storage_t build_scene(const std::vector<drawable_original_transform> &drawables_orig_transfo, nova::NovaResourceManager &manager) {
    core::memory::ByteArena &memory_pool = manager.getMemoryPool();
    std::vector<Mesh *> meshes = retrieve_meshes_from_drawables(drawables_orig_transfo);
    /* Allocate for triangles */
    std::size_t primitive_number = compute_primitive_number(meshes);
    primitive_buffers_t primitive_buffers = allocate_primitive_triangle_buffers(memory_pool, primitive_number);

    /* Allocate one singular contiguous buffer of ImageTexture objects*/
    auto *image_texture_buffer = reinterpret_cast<nova::texturing::ImageTexture *>(
        memory_pool.allocate(PBR_PIPELINE_TEX_NUM * meshes.size() * sizeof(nova::texturing::ImageTexture), "ImageTexture buffer"));
    texture_buffers_t texture_buffers{};
    texture_buffers.image_alloc_buffer = axstd::span<nova::texturing::ImageTexture>(image_texture_buffer, PBR_PIPELINE_TEX_NUM * meshes.size());
    material_buffers_t material_buffers = allocate_materials_buffers(manager.getMemoryPool(), meshes.size());
    initialize_resources_holders(manager, meshes.size());
    std::size_t alloc_offset_primitives = 0, alloc_offset_materials = 0, alloc_offset_textures = 0, mesh_index = 0;
    for (const drawable_original_transform &dtf : drawables_orig_transfo) {

      nova::material::NovaMaterialInterface material = setup_material_data(
          material_buffers, texture_buffers, *dtf.mesh, manager, alloc_offset_textures, alloc_offset_materials);

      setup_geometry_data(primitive_buffers, dtf, alloc_offset_primitives, material, manager, mesh_index);
      mesh_index++;
    }
    manager.getShapeData().updateSharedBuffers();
    bake_buffers_storage_t bake_buffers_storage{};
    bake_buffers_storage.material_buffers = material_buffers;
    bake_buffers_storage.primitive_buffers = primitive_buffers;
    bake_buffers_storage.texture_buffers = texture_buffers;
    return bake_buffers_storage;
  }

  nova::aggregate::Accelerator build_performance_acceleration_structure(const axstd::span<nova::primitive::NovaPrimitiveInterface> &primitives) {
    nova::aggregate::Accelerator accelerator{};
    accelerator.buildBVH(primitives);
    return accelerator;
  }

  nova::aggregate::Accelerator build_quality_acceleration_structure(const axstd::span<nova::primitive::NovaPrimitiveInterface> &primitives) {
    nova::aggregate::Accelerator accelerator{};
    accelerator.buildBVH(primitives);
    return accelerator;
  }

}  // namespace nova_baker_utils
