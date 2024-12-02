#include "INodeDatabase.h"
#include "INodeFactory.h"
#include "Loader.h"
#include "ResourceDatabaseManager.h"
#include "ShaderDatabase.h"
#include "TextureDatabase.h"
#include "internal/common/string/string_utils.h"

#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>

// TODO : code needs total uncoupling from any foreign structure : needs to return arrays of basic data

namespace IO {
  namespace exception {}
  Loader::Loader(controller::ProgressStatus *prog_stat) : resource_database(&ResourceDatabaseManager::getInstance()) {
    progress_manager.setProgressManager(prog_stat);
  }

  static loader_data_t loadSceneElements(const aiScene *modelScene,
                                         ShaderDatabase *shader_database,
                                         TextureDatabase *texture_database,
                                         INodeDatabase *node_database,
                                         controller::IProgressManager &progress_manager) {
    std::vector<GLMaterial> material_array;
    std::vector<Mesh *> node_lookup_table;
    Shader *shader_program = shader_database->get(Shader::BRDF);
    node_lookup_table.resize(modelScene->mNumMeshes);
    material_array.resize(modelScene->mNumMeshes);
    loader_data_t scene_data;
    std::size_t texture_cache_size = total_textures_size(modelScene);
    std::size_t geometry_cache_size = total_geometry_size(modelScene);
    uint8_t *texture_cache_address = texture_database->reserveCache(texture_cache_size, core::memory::PLATFORM_ALIGN);
    uint8_t *geometry_cache_address = node_database->reserveCache(geometry_cache_size, core::memory::PLATFORM_ALIGN);
    scene_data.texture_cache = axstd::span<uint8_t>(texture_cache_address, texture_cache_size);
    scene_data.geometry_cache = axstd::span<uint8_t>(geometry_cache_address, geometry_cache_size);
    std::size_t texcache_element_count = 0, geocache_element_count = 0;
    for (unsigned int mesh_index = 0; mesh_index < modelScene->mNumMeshes; mesh_index++) {

      std::pair<unsigned, std::unique_ptr<Object3D>> geometry_loaded = load_geometry(
          modelScene, mesh_index, *node_database, geocache_element_count, progress_manager);

      unsigned int mMaterialIndex = modelScene->mMeshes[mesh_index]->mMaterialIndex;
      material_array[mesh_index] = load_materials(modelScene, mMaterialIndex, *texture_database, texcache_element_count, progress_manager).second;

      const aiMesh *mesh = modelScene->mMeshes[mesh_index];
      const char *mesh_name = mesh->mName.C_Str();
      std::string name(mesh_name);
      auto mesh_result = database::node::store<Mesh>(
          *node_database, false, name, std::move(*geometry_loaded.second), material_array[mesh_index], shader_program, nullptr);
      LOG("object loaded : " + name, LogLevel::INFO);
      node_lookup_table[mesh_index] = mesh_result.object;
      scene_data.mesh_list.push_back(mesh_result.object);
    }
    SceneTree scene_tree = generateSceneTree(modelScene, node_lookup_table);
    scene_data.scene_tree = scene_tree;
    return scene_data;
  }

  loader_data_t Loader::loadObjects(const char *file) {
    ShaderDatabase *shader_database = resource_database->getShaderDatabase();
    INodeDatabase *node_database = resource_database->getNodeDatabase();
    TextureDatabase *texture_database = resource_database->getTextureDatabase();
    loader_data_t loader_data;
    if (!texture_database) {
      LOG("Texture database is not initialized.", LogLevel::CRITICAL);
      return loader_data;
    }
    if (!node_database) {
      LOG("Node database is not initialized.", LogLevel::CRITICAL);
      return loader_data;
    }
    if (!shader_database) {
      LOG("Shader database is not initialized.", LogLevel::CRITICAL);
      return loader_data;
    }

    Assimp::Importer importer;
    const aiScene *modelScene = importer.ReadFile(
        file, aiProcess_CalcTangentSpace | aiProcess_Triangulate | aiProcess_JoinIdenticalVertices | aiProcess_FlipUVs);
    if (modelScene) {
      return loadSceneElements(modelScene, shader_database, texture_database, node_database, progress_manager);
    } else {
      LOG("Cannot read file.", LogLevel::ERROR);
      return loader_data;
    }
  }

  /**
   * The function loads a file and generates a vector of meshes, including a cube map if applicable.
   * @param file A pointer to a character array representing the file path of the 3D model to be loaded.
   * @return A vector of Mesh pointers.
   */
  loader_data_t Loader::load(const char *file) {
    loader_data_t scene = loadObjects(file);
    return scene;
  }

}  // namespace IO
