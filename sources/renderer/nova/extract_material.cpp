#include "Drawable.h"
#include "MaterialInterface.h"
#include "Mesh.h"
#include "bake.h"
#include "texturing/nova_texturing.h"
#include <internal/common/exception/GenericException.h>

// TODO : this could be replaced with cuda<->opengl interop
namespace exception {
  class InvalidTexTypeConversionException : public CatastrophicFailureException {
   public:
    InvalidTexTypeConversionException() : CatastrophicFailureException() { saveErrorString("Invalid tag pointer conversion."); }
  };
}  // namespace exception

namespace nova_baker_utils {
  nova::texturing::ImageTexture *extract_texture(nova::texturing::ImageTexture *alloc_buffer,
                                                 std::size_t offset,
                                                 const TextureGroup &tgroup,
                                                 nova::NovaResourceManager &manager,
                                                 GenericTexture::TYPE type) {
    const GenericTexture *gltexture = tgroup.getTexturePointer(type);
    if (!gltexture) {
      LOG("Texture lookup in Nova scene initialization has returned null.", LogLevel::WARNING);
      return nullptr;
    }
    const auto *buffer_ptr = gltexture->getData();
    int w = (int)gltexture->getWidth();
    int h = (int)gltexture->getHeight();
    auto ret = manager.getTexturesData().add_texture<nova::texturing::ImageTexture>(
        manager.getMemoryPool(), alloc_buffer, offset, buffer_ptr, w, h, 4);
    auto *image_tex = ret.get<nova::texturing::ImageTexture>();
    if (!image_tex)
      throw exception::InvalidTexTypeConversionException();
    return image_tex;
  }

  /* Will be removed when BSDFs will be implemented */
  material_buffers_t allocate_materials_buffers(core::memory::ByteArena &memory_pool, std::size_t number_elements) {
    material_buffers_t buffers{};
    auto *alloc_diffuse = reinterpret_cast<nova::material::NovaDiffuseMaterial *>(
        memory_pool.allocate(number_elements * sizeof(nova::material::NovaDiffuseMaterial), "NovaDiffuseMaterial buffer"));
    auto *alloc_dielec = reinterpret_cast<nova::material::NovaDielectricMaterial *>(
        memory_pool.allocate(number_elements * sizeof(nova::material::NovaDielectricMaterial), "NovaDielectricMaterial buffer"));
    auto *alloc_conduc = reinterpret_cast<nova::material::NovaConductorMaterial *>(
        memory_pool.allocate(number_elements * sizeof(nova::material::NovaConductorMaterial), "NovaConductorMaterial buffer"));
    AX_ASSERT_NOTNULL(alloc_diffuse);
    AX_ASSERT_NOTNULL(alloc_dielec);
    AX_ASSERT_NOTNULL(alloc_conduc);
    buffers.diffuse_alloc_buffer = axstd::span<nova::material::NovaDiffuseMaterial>(alloc_diffuse, number_elements);
    buffers.dielectric_alloc_buffer = axstd::span<nova::material::NovaDielectricMaterial>(alloc_dielec, number_elements);
    buffers.conductor_alloc_buffer = axstd::span<nova::material::NovaConductorMaterial>(alloc_conduc, number_elements);
    return buffers;
  }

  static nova::material::texture_pack extract_materials(texture_buffers_t &texture_buffers,
                                                        std::size_t offset,
                                                        const Mesh *mesh,
                                                        nova::NovaResourceManager &manager) {
    const MaterialInterface *material = mesh->getMaterial();
    const TextureGroup &texture_group = material->getTextureGroup();
    nova::material::texture_pack tpack{};
    auto *allocated_texture_buffer = texture_buffers.image_alloc_buffer.data();
    tpack.albedo = extract_texture(allocated_texture_buffer, offset, texture_group, manager, GenericTexture::DIFFUSE);
    tpack.metallic = extract_texture(allocated_texture_buffer, offset + 1, texture_group, manager, GenericTexture::METALLIC);
    tpack.normalmap = extract_texture(allocated_texture_buffer, offset + 2, texture_group, manager, GenericTexture::NORMAL);
    tpack.roughness = extract_texture(allocated_texture_buffer, offset + 3, texture_group, manager, GenericTexture::ROUGHNESS);
    tpack.ao = extract_texture(allocated_texture_buffer, offset + 4, texture_group, manager, GenericTexture::AMBIANTOCCLUSION);
    tpack.emissive = extract_texture(allocated_texture_buffer, offset + 5, texture_group, manager, GenericTexture::EMISSIVE);
    return tpack;
  }

  // For now it assigns materials randomly, I don't want to waste time with a proper material translation system since I'm gonna scrape it for a more
  // uniform pipeline with PBR
  static nova::material::NovaMaterialInterface assign_random_material(material_buffers_t &material_buffers,
                                                                      nova::material::texture_pack &tpack,
                                                                      nova::NovaResourceManager &manager,
                                                                      std::size_t &offset) {
    math::random::CPUPseudoRandomGenerator rand_gen;
    int r = rand_gen.nrandi(0, 2);
    nova::material::NovaMaterialInterface mat_ptr{};
    core::memory::ByteArena &arena = manager.getMemoryPool();
    switch (r) {
      case 0:
        mat_ptr = manager.getMaterialData().add_material<nova::material::NovaConductorMaterial>(
            arena, material_buffers.conductor_alloc_buffer.data(), offset, tpack, rand_gen.nrandf(0.001, 0.5));
        break;
      case 1:
        mat_ptr = manager.getMaterialData().add_material<nova::material::NovaDielectricMaterial>(
            arena, material_buffers.dielectric_alloc_buffer.data(), offset, tpack, rand_gen.nrandf(1.5, 2.4));
        break;
      case 2:
        mat_ptr = manager.getMaterialData().add_material<nova::material::NovaDiffuseMaterial>(
            arena, material_buffers.diffuse_alloc_buffer.data(), offset, tpack);
        break;
      default:
        mat_ptr = manager.getMaterialData().add_material<nova::material::NovaConductorMaterial>(
            arena, material_buffers.conductor_alloc_buffer.data(), offset, tpack, 0.004);
        break;
    }
    return mat_ptr;
  }

  nova::material::NovaMaterialInterface setup_material_data(material_buffers_t &material_buffers,
                                                            texture_buffers_t &texture_buffer,
                                                            const Drawable &drawable,
                                                            nova::NovaResourceManager &manager,
                                                            std::size_t &alloc_offset_textures,
                                                            std::size_t &alloc_offset_materials) {
    const Mesh *mesh = drawable.getMeshPointer();
    const MaterialInterface *material = mesh->getMaterial();
    if (!material) {
      return nullptr;
    }
    nova::material::texture_pack tpack = extract_materials(texture_buffer, alloc_offset_textures, mesh, manager);
    nova::material::NovaMaterialInterface mat = assign_random_material(material_buffers, tpack, manager, alloc_offset_materials);
    alloc_offset_textures = alloc_offset_textures + PBR_PIPELINE_TEX_NUM;
    alloc_offset_materials += 1;
    return mat;
  }
}  // namespace nova_baker_utils