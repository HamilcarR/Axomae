#include "MaterialInterface.h"
#include "Mesh.h"
#include "TextureGroup.h"
#include "bake.h"
#include "internal/common/axstd/span.h"
#include "internal/macro/project_macros.h"
#include "manager/NovaResourceManager.h"
#include "material/nova_material.h"
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
    auto *triangle_buffer = memory_pool.construct<nova::shape::Triangle>(primitive_number, false);
    auto *primitive_buffer = memory_pool.construct<nova::primitive::NovaGeoPrimitive>(primitive_number, false);
    primitive_buffers_t geometry_buffers{};
    geometry_buffers.geo_primitive_alloc_buffer = primitive_buffer;
    geometry_buffers.triangle_alloc_buffer = triangle_buffer;
    return geometry_buffers;
  }

  /**
   * Each mesh has PBR_PIPELINE_TEX_NUM number of textures
   */
  template<class T>
  T *allocate_type_texture_buffer(core::memory::ByteArena &memory_arena,
                                  std::size_t num_textures,
                                  std::size_t alignment = core::memory::PLATFORM_ALIGN) {
    return memory_arena.allocate(sizeof(T) * num_textures, alignment);
  }

  bake_buffers_storage_t build_scene(const std::vector<Mesh *> &meshes, nova::NovaResourceManager &manager) {
    core::memory::ByteArena &memory_pool = manager.getMemoryPool();

    /* Allocate for triangles */
    std::size_t primitive_number = compute_primitive_number(meshes);
    primitive_buffers_t primitive_buffers = allocate_primitive_triangle_buffers(manager.getMemoryPool(), primitive_number);

    /* Allocate one singular contiguous buffer of ImageTexture objects*/
    auto *image_texture_buffer = reinterpret_cast<uint8_t *>(
        memory_pool.construct<nova::texturing::ImageTexture>(PBR_PIPELINE_TEX_NUM * meshes.size(), true));
    texture_buffers_t texture_buffers{};
    texture_buffers.image_alloc_buffer = axstd::span<uint8_t>(image_texture_buffer, PBR_PIPELINE_TEX_NUM * meshes.size());
    material_buffers_t material_buffers = allocate_materials_buffers(manager.getMemoryPool(), meshes.size());
    std::size_t alloc_offset_primitives = 0, alloc_offset_materials = 0;

    for (const auto &elem : meshes) {
      nova::material::NovaMaterialInterface material = setup_material_data(material_buffers, texture_buffers, elem, manager, alloc_offset_materials);
      setup_geometry_data(primitive_buffers, elem, alloc_offset_primitives, material, manager);
    }
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
