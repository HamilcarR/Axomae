#include "Loader.h"
#include "TextureDatabase.h"
#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>

/***********************************************************************************************************************************************/
/* Materials and textures */
namespace IO {
  /**
   * The function copies texture data from a GLB file to an ARGB8888 buffer.
   *
   * @param totexture A pointer to a TextureData struct that will hold the copied texture data.
   * @param fromtexture The aiTexture object containing the texture data to be copied.
   */
  static void copyTexels(U32TexData *totexture,
                         aiTexture *fromtexture,
                         const std::string &texture_type,
                         TextureDatabase &texture_database,
                         std::size_t &texcache_element_count) {

    if (fromtexture != nullptr) {
      /* If mHeight != 0 , the texture is uncompressed , and we read it as is */
      if (fromtexture->mHeight != 0) {
        unsigned int width = 0;
        unsigned int height = 0;
        totexture->width = width = fromtexture->mWidth;
        totexture->height = height = fromtexture->mHeight;
        std::vector<uint32_t> temp_texdata;
        temp_texdata.resize(width * height);
        for (unsigned int i = 0; i < width * height; i++) {
          uint8_t a = fromtexture->pcData[i].a;
          uint8_t r = fromtexture->pcData[i].r;
          uint8_t g = fromtexture->pcData[i].g;
          uint8_t b = fromtexture->pcData[i].b;
          uint32_t rgba = (a << 24) | (b << 16) | (g << 8) | r;
          temp_texdata[i] = rgba;
        }
        totexture->data = texture_database.copyRangeToCache(temp_texdata.data(), nullptr, width * height, texcache_element_count);
        AX_ASSERT_NOTNULL(totexture->data);
        texcache_element_count += 1;
      }
      /* If mHeight = 0 , the texture is compressed , and we need to uncompress and convert it to ARGB32 */
      else
      {
        QImage image;
        image.loadFromData(reinterpret_cast<const uint8_t *>(fromtexture->pcData), static_cast<int>(fromtexture->mWidth));
        image = image.convertToFormat(QImage::Format_ARGB32);
        unsigned image_width = image.width();
        unsigned image_height = image.height();
        uint32_t *from_buffer = reinterpret_cast<uint32_t *>(image.bits());
        totexture->data = texture_database.copyRangeToCache(from_buffer, nullptr, image_width * image_height, texcache_element_count);
        AX_ASSERT_NOTNULL(totexture->data);
        texcache_element_count += image_width * image_height;
        totexture->width = image_width;
        totexture->height = image_height;
        LOG("image of size " + std::to_string(totexture->width) + " x " + std::to_string(totexture->height) + " uncompressed ", LogLevel::INFO);
      }
    }
  }

  template<class TEXTYPE>
  static void loadTextureDummy(GLMaterial *material, TextureDatabase &texture_database) {
    auto result = database::texture::store<TEXTYPE>(texture_database, true, nullptr);
    LOG("Loading dummy texture at index : " + std::to_string(result.id), LogLevel::INFO);
    material->addTexture(result.id);
  }

  template<class TEXTYPE>
  static void loadTexture(const aiScene *scene,
                          GLMaterial *material,
                          U32TexData &texture,
                          const aiString &texture_string,
                          TextureDatabase &texture_database,
                          std::size_t &texcache_element_count) {
    std::string texture_index_string = texture_string.C_Str();
    std::string texture_type = texture.name;
    if (texture_index_string.empty()) {
      LOG("Loader failed loading texture of type: " + texture_type, LogLevel::ERROR);
      return;
    }
    /*Get rid of the '*' character at the beginning of the string id*/
    texture_index_string = texture_index_string.substr(1);
    texture.name = texture_index_string;
    /*Check if the name (the assimp texture id number) is not present in the database. (avoids duplicate) */
    auto result = texture_database.getUniqueTexture(texture.name);
    if (!result.object) {
      /*Convert id to integer*/
      int texture_index_int = stoi(texture_index_string);
      /*Read the image pixels and copy them to "texture"*/
      copyTexels(&texture, scene->mTextures[texture_index_int], texture_type, texture_database, texcache_element_count);
      /*Add new texture*/
      auto result_add = database::texture::store<TEXTYPE>(texture_database, false, &texture);
      material->addTexture(result_add.id);
    } else {
      material->addTexture(result.id);
    }
  }

  /**
   * The function loads textures for a given material in a 3D model scene.
   * @param scene a pointer to the aiScene object which contains the loaded 3D model data.
   * @param material The aiMaterial object that contains information about the material properties of a
   * 3D model.
   * @return a Material object.
   */
  static GLMaterial loadAllTextures(const aiScene *scene,
                                    const aiMaterial *material,
                                    TextureDatabase &texture_database,
                                    std::size_t &texcache_element_count) {
    GLMaterial mesh_material;
    std::vector<GenericTexture::TYPE> dummy_textures_type;
    U32TexData diffuse, metallic, roughness, normal, ambiantocclusion, emissive, specular, opacity;
    diffuse.name = "diffuse";
    metallic.name = "metallic";
    roughness.name = "roughness";
    opacity.name = "opacity";
    normal.name = "normal";
    ambiantocclusion.name = "occlusion";
    specular.name = "specular";
    emissive.name = "emissive";
    aiString color_texture, opacity_texture, normal_texture, metallic_texture, roughness_texture, emissive_texture, specular_texture,
        occlusion_texture;
    if (material->GetTextureCount(aiTextureType_BASE_COLOR) > 0) {
      material->GetTexture(AI_MATKEY_BASE_COLOR_TEXTURE, &color_texture);
      loadTexture<DiffuseTexture>(scene, &mesh_material, diffuse, color_texture, texture_database, texcache_element_count);
    } else
      loadTextureDummy<DiffuseTexture>(&mesh_material, texture_database);

    if (material->GetTextureCount(aiTextureType_OPACITY) > 0) {
      mesh_material.setAlphaFactor(true);
      material->GetTexture(aiTextureType_OPACITY, 0, &opacity_texture, nullptr, nullptr, nullptr, nullptr, nullptr);
      loadTexture<OpacityTexture>(scene, &mesh_material, opacity, opacity_texture, texture_database, texcache_element_count);
    } else
      loadTextureDummy<OpacityTexture>(&mesh_material, texture_database);

    if (material->GetTextureCount(aiTextureType_METALNESS) > 0) {
      material->GetTexture(AI_MATKEY_METALLIC_TEXTURE, &metallic_texture);
      loadTexture<MetallicTexture>(scene, &mesh_material, metallic, metallic_texture, texture_database, texcache_element_count);
    } else
      loadTextureDummy<MetallicTexture>(&mesh_material, texture_database);

    if (material->GetTextureCount(aiTextureType_DIFFUSE_ROUGHNESS) > 0) {
      material->GetTexture(AI_MATKEY_ROUGHNESS_TEXTURE, &roughness_texture);
      loadTexture<RoughnessTexture>(scene, &mesh_material, roughness, roughness_texture, texture_database, texcache_element_count);
    } else
      loadTextureDummy<RoughnessTexture>(&mesh_material, texture_database);

    if (material->GetTextureCount(aiTextureType_NORMALS) > 0) {
      material->GetTexture(aiTextureType_NORMALS, 0, &normal_texture, nullptr, nullptr, nullptr, nullptr, nullptr);
      loadTexture<NormalTexture>(scene, &mesh_material, normal, normal_texture, texture_database, texcache_element_count);
    } else
      loadTextureDummy<NormalTexture>(&mesh_material, texture_database);

    if (material->GetTextureCount(aiTextureType_LIGHTMAP) > 0) {
      material->GetTexture(aiTextureType_LIGHTMAP, 0, &occlusion_texture, nullptr, nullptr, nullptr, nullptr, nullptr);
      loadTexture<AmbiantOcclusionTexture>(scene, &mesh_material, ambiantocclusion, occlusion_texture, texture_database, texcache_element_count);
    } else
      loadTextureDummy<AmbiantOcclusionTexture>(&mesh_material, texture_database);

    if (material->GetTextureCount(aiTextureType_SHEEN) > 0) {
      material->GetTexture(aiTextureType_SHEEN, 0, &specular_texture, nullptr, nullptr, nullptr, nullptr, nullptr);
      loadTexture<SpecularTexture>(scene, &mesh_material, specular, specular_texture, texture_database, texcache_element_count);
    } else
      loadTextureDummy<SpecularTexture>(&mesh_material, texture_database);

    if (material->GetTextureCount(aiTextureType_EMISSIVE) > 0) {
      material->GetTexture(aiTextureType_EMISSIVE, 0, &emissive_texture, nullptr, nullptr, nullptr, nullptr, nullptr);
      loadTexture<EmissiveTexture>(scene, &mesh_material, emissive, emissive_texture, texture_database, texcache_element_count);
    } else
      loadTextureDummy<EmissiveTexture>(&mesh_material, texture_database);

    return mesh_material;
  }

  static float loadTransparencyValue(const aiMaterial *material) {
    float transparency = 1.f;
    float opacity = 1.f;
    aiColor4D col;
    if (material->Get(AI_MATKEY_COLOR_TRANSPARENT, col) == AI_SUCCESS)
      transparency = col.a;
    else if (material->Get(AI_MATKEY_OPACITY, opacity) == AI_SUCCESS)
      transparency = 1.f - opacity;
    return transparency;
  }

  static float loadRefractiveValue(const aiMaterial *material) {
    float ior = 0.f;
    material->Get(AI_MATKEY_REFRACTI, ior);
    return ior;
  }

  /* id will be incremented by the texture caching system*/
  std::pair<unsigned, GLMaterial> load_materials(const aiScene *scene,
                                                 unsigned mMaterialIndex,
                                                 TextureDatabase &texture_database,
                                                 std::size_t &texcache_element_count) {
    const aiMaterial *material = scene->mMaterials[mMaterialIndex];
    GLMaterial mesh_material = loadAllTextures(scene, material, texture_database, texcache_element_count);

    float transparency_factor = loadTransparencyValue(material);
    mesh_material.setAlphaFactor(transparency_factor);

    float ior = loadRefractiveValue(material);
    mesh_material.setRefractiveIndexValue(ior, 0.f);

    return {texcache_element_count, mesh_material};
  }

  std::size_t total_textures_size(const aiScene *scene, controller::IProgressManager *progress_manager) {
    std::size_t total_size = 0;
    progress_manager->initProgress("Computing texture cache size ", static_cast<float>(scene->mNumTextures));
    for (unsigned int i = 0; i < scene->mNumTextures; ++i) {
      aiTexture *texture = scene->mTextures[i];
      if (texture->mHeight == 0) {
        QImage image;
        image.loadFromData(reinterpret_cast<uint8_t *>(texture->pcData), static_cast<int>(texture->mWidth));
        image = image.convertToFormat(QImage::Format_ARGB32);
        total_size += image.width() * image.height() * sizeof(uint32_t);
      } else {
        total_size += texture->mWidth * texture->mHeight * sizeof(uint32_t);
      }
      progress_manager->setCurrent((float)i);
      progress_manager->notifyProgress();
    }
    progress_manager->resetProgress();
    return total_size;
  }
}  // namespace IO