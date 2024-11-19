#include "MaterialInterface.h"
#include "Mesh.h"
#include "bake.h"
#include "texturing/nova_texturing.h"
#include <internal/common/exception/GenericException.h>
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

    buffers.diffuse_alloc_buffer = memory_pool.construct<nova::material::NovaDiffuseMaterial>(number_elements, false);
    buffers.conductor_alloc_buffer = memory_pool.construct<nova::material::NovaConductorMaterial>(number_elements, false);
    buffers.dielectric_alloc_buffer = memory_pool.construct<nova::material::NovaDielectricMaterial>(number_elements, false);
    return buffers;
  }

  nova::material::texture_pack extract_materials(texture_buffers_t &texture_buffers,
                                                 std::size_t offset,
                                                 const Mesh *mesh,
                                                 nova::NovaResourceManager &manager) {
    const MaterialInterface *material = mesh->getMaterial();
    const TextureGroup &texture_group = material->getTextureGroup();
    nova::material::texture_pack tpack{};
    auto *allocated_texture_buffer = reinterpret_cast<nova::texturing::ImageTexture *>(texture_buffers.image_alloc_buffer.data());
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
    int r = math::random::nrandi(0, 1);
    nova::material::NovaMaterialInterface mat_ptr{};
    switch (r) {
      case 0:
        mat_ptr = manager.getMaterialData().add_material<nova::material::NovaConductorMaterial>(
            material_buffers.conductor_alloc_buffer, offset, tpack, math::random::nrandf(0.001, 0.5));
        break;
      case 1:
        mat_ptr = manager.getMaterialData().add_material<nova::material::NovaDielectricMaterial>(
            material_buffers.dielectric_alloc_buffer, offset, tpack, math::random::nrandf(1.5, 2.4));
        break;
      case 2:
        mat_ptr = manager.getMaterialData().add_material<nova::material::NovaDiffuseMaterial>(material_buffers.diffuse_alloc_buffer, offset, tpack);
        break;
      default:
        mat_ptr = manager.getMaterialData().add_material<nova::material::NovaConductorMaterial>(
            material_buffers.conductor_alloc_buffer, offset, tpack, 0.004);
        break;
    }
    return mat_ptr;
  }

  nova::material::NovaMaterialInterface setup_material_data(material_buffers_t &material_buffers,
                                                            texture_buffers_t &texture_buffer,
                                                            const Mesh *mesh,
                                                            nova::NovaResourceManager &manager,
                                                            std::size_t &alloc_offset_materials) {
    const MaterialInterface *material = mesh->getMaterial();
    if (!material) {
      return nullptr;
    }
    nova::material::texture_pack tpack = extract_materials(texture_buffer, alloc_offset_materials, mesh, manager);
    nova::material::NovaMaterialInterface mat = assign_random_material(material_buffers, tpack, manager, alloc_offset_materials);
    alloc_offset_materials = alloc_offset_materials + PBR_PIPELINE_TEX_NUM;
    return mat;
  }
}  // namespace nova_baker_utils