#include "INodeDatabase.h"
#include "INodeFactory.h"
#include "Loader.h"
#include "ResourceDatabaseManager.h"
#include "ShaderDatabase.h"
#include "TextureDatabase.h"
#include "internal/common/string/string_utils.h"
#include <QImage>

#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>

// TODO : code needs total uncoupling from any foreign structure : needs to return arrays of basic data

namespace IO {
  namespace exception {}

  Loader::Loader(controller::ProgressStatus *prog_stat) : resource_database(&ResourceDatabaseManager::getInstance()) {
    progress_manager.setProgressManager(prog_stat);
  }

  /***********************************************************************************************************************************************/
  /* Geometry */
  /**
   * The function `aiMatrix4x4ToGlm` converts an `aiMatrix4x4` object to a `glm::mat4` object in C++.
   *
   * @param from The "from" parameter is of type aiMatrix4x4, which is a 4x4 matrix structure used in the
   * Assimp library. It represents a transformation matrix.
   *
   * @return a glm::mat4 object.
   */
  inline glm::mat4 aiMatrix4x4ToGlm(const aiMatrix4x4 &from) {  // https://stackoverflow.com/a/29184538
    glm::mat4 to;
    to[0][0] = (GLfloat)from.a1;
    to[0][1] = (GLfloat)from.b1;
    to[0][2] = (GLfloat)from.c1;
    to[0][3] = (GLfloat)from.d1;
    to[1][0] = (GLfloat)from.a2;
    to[1][1] = (GLfloat)from.b2;
    to[1][2] = (GLfloat)from.c2;
    to[1][3] = (GLfloat)from.d2;
    to[2][0] = (GLfloat)from.a3;
    to[2][1] = (GLfloat)from.b3;
    to[2][2] = (GLfloat)from.c3;
    to[2][3] = (GLfloat)from.d3;
    to[3][0] = (GLfloat)from.a4;
    to[3][1] = (GLfloat)from.b4;
    to[3][2] = (GLfloat)from.c4;
    to[3][3] = (GLfloat)from.d4;
    return to;
  }

  SceneTreeNode *fillTreeData(aiNode *ai_node, const std::vector<Mesh *> &mesh_lookup, SceneTreeNode *parent) {
    INodeDatabase *node_database = ResourceDatabaseManager::getInstance().getNodeDatabase();
    if (ai_node != nullptr) {
      std::string name = ai_node->mName.C_Str();
      glm::mat4 transformation = aiMatrix4x4ToGlm(ai_node->mTransformation);
      std::vector<SceneTreeNode *> add_node;
      if (ai_node->mNumMeshes == 0) {
        SceneTreeNode *empty_node =
            database::node::store<SceneTreeNode>(*node_database, false, parent).object;  //* Empty node , no other goal than transformation node .
        add_node.push_back(empty_node);
      } else if (ai_node->mNumMeshes == 1)
        add_node.push_back(mesh_lookup[ai_node->mMeshes[0]]);
      else {
        /* Little compatibility hack between assimp and the node system, assimp
         nodes can contain multiples meshes , but SceneTreeNode can be a mesh.
        So we create a dummy node at position 0 in add_node to be the ancestors of the children
        nodes , while meshes will be attached to parent and without children.*/
        database::node::store<SceneTreeNode>(*node_database, false, parent);
        for (unsigned i = 0; i < ai_node->mNumMeshes; i++)
          add_node.push_back(mesh_lookup[ai_node->mMeshes[i]]);
      }
      for (auto A : add_node) {
        A->setLocalModelMatrix(transformation);
        A->setName(name);
        std::vector<datastructure::NodeInterface *> parents_array = {parent};
        A->setParents(parents_array);
      }
      for (unsigned i = 0; i < ai_node->mNumChildren; i++)
        fillTreeData(ai_node->mChildren[i], mesh_lookup, add_node[0]);
      return add_node[0];
    }
    return nullptr;
  }

  /**
   * The function generates a scene tree from a model scene and a lookup of mesh nodes.
   * @param modelScene The modelScene parameter is a pointer to an aiScene object, which represents a 3D
   * model scene loaded from a file using the Assimp library. It contains information about the model's
   * hierarchy, meshes, materials, textures, animations, etc.
   * @param node_lookup The `node_lookup` parameter is a vector of pointers to `Mesh` objects. It is used
   * to map the `aiNode` objects in the `modelScene` to their corresponding `Mesh` objects.
   * @return a SceneTree object.
   */
  SceneTree generateSceneTree(const aiScene *modelScene, const std::vector<Mesh *> &node_lookup) {
    aiNode *ai_root = modelScene->mRootNode;
    SceneTree scene_tree;
    SceneTreeNode *node = fillTreeData(ai_root, node_lookup, nullptr);
    AX_ASSERT_NEQ(node, nullptr);
    node = dynamic_cast<SceneTreeNode *>(node->returnRoot());
    scene_tree.setRoot(node);
    scene_tree.updateAccumulatedTransformations();
    return scene_tree;
  }

  /**
   * The function "load_geometry_buffer" takes an array of aiVector3D objects, converts them to a
   * specified dimension, and stores the result in a destination vector.
   * @param dest The `dest` parameter is a reference to a `std::vector<T>` object, where `T` is the type
   * of the elements in the vector. This vector will be used to store the loaded geometry data.
   * @param from The "from" parameter is a pointer to an array of aiVector3D objects.
   * @param size The size parameter represents the number of elements in the from array.
   * @param dimension The "dimension" parameter determines the number of components in each vector. If
   * dimension is set to 3, each vector will have three components (x, y, and z). If dimension is set to
   * 2, each vector will have two components (x and y).
   */
  template<class T>
  void load_geometry_buffer(std::vector<T> &dest, const aiVector3D *from, int size, int dimension) {
    for (int f = 0, i = 0; f < size; f++) {
      const aiVector3D vect = from[f];
      if (dimension == 3) {
        dest[i] = vect.x;
        dest[i + 1] = vect.y;
        dest[i + 2] = vect.z;
        i += 3;
      } else {
        dest[i] = vect.x;
        dest[i + 1] = vect.y;
        i += 2;
      }
    }
  }

  /**
   * The function `load_indices_buffer` loads the indices of a mesh's faces into a destination vector.
   * @param dest A reference to a vector of unsigned integers where the indices will be loaded into.
   * @param faces The `faces` parameter is a pointer to an array of `aiFace` objects. Each `aiFace`
   * object represents a face in a 3D model and contains an array of indices that define the vertices of
   * the face.
   * @param num_faces The parameter "num_faces" represents the number of faces in the mesh.
   */
  void load_indices_buffer(std::vector<unsigned> &dest, const aiFace *faces, int num_faces) {
    for (int i = 0, f = 0; i < num_faces; i++, f += 3) {
      assert(faces[i].mNumIndices == 3);
      dest[f] = faces[i].mIndices[0];
      dest[f + 1] = faces[i].mIndices[1];
      dest[f + 2] = faces[i].mIndices[2];
    }
  }

  /**
   * This function loads vertex attributes in an async manner.
   */
  static std::pair<unsigned, std::unique_ptr<Object3D>> retrieve_geometry(const aiScene *modelScene, unsigned i) {
    const aiMesh *mesh = modelScene->mMeshes[i];
    AX_ASSERT_NOTNULL(mesh->mVertices);

    std::unique_ptr<Object3D> mesh_geometry_buffers = std::make_unique<Object3D>();
    auto size_dim3 = mesh->mNumVertices * 3;
    auto size_dim2 = mesh->mNumVertices * 2;
    auto size_dim3_indices = mesh->mNumFaces * 3;

    std::vector<std::shared_future<void>> shared_futures;
    std::future<void> f_vertices, f_normals, f_bitangents, f_tangents, f_uv;
    mesh_geometry_buffers->vertices.resize(size_dim3);
    f_vertices = std::async(std::launch::async,
                            [&]() { load_geometry_buffer(mesh_geometry_buffers->vertices, mesh->mVertices, mesh->mNumVertices, 3); });
    shared_futures.push_back(f_vertices.share());

    if (mesh->HasNormals()) {
      mesh_geometry_buffers->normals.resize(size_dim3);
      f_normals = std::async(std::launch::async,
                             [&]() { load_geometry_buffer(mesh_geometry_buffers->normals, mesh->mNormals, mesh->mNumVertices, 3); });
      shared_futures.push_back(f_normals.share());
    }

    if (mesh->HasTextureCoords(0)) {
      mesh_geometry_buffers->uv.resize(size_dim2);
      f_uv = std::async(std::launch::async,
                        [&]() { load_geometry_buffer(mesh_geometry_buffers->uv, mesh->mTextureCoords[0], mesh->mNumVertices, 2); });
      shared_futures.push_back(f_uv.share());
    }

    if (mesh->HasTangentsAndBitangents()) {
      mesh_geometry_buffers->tangents.resize(size_dim3);
      mesh_geometry_buffers->bitangents.resize(size_dim3);
      f_tangents = std::async(std::launch::async,
                              [&]() { load_geometry_buffer(mesh_geometry_buffers->tangents, mesh->mTangents, mesh->mNumVertices, 3); });
      f_bitangents = std::async(std::launch::async,
                                [&]() { load_geometry_buffer(mesh_geometry_buffers->bitangents, mesh->mBitangents, mesh->mNumVertices, 3); });
      shared_futures.push_back(f_tangents.share());
      shared_futures.push_back(f_bitangents.share());
    }

    mesh_geometry_buffers->indices.resize(size_dim3_indices);
    load_indices_buffer(mesh_geometry_buffers->indices, mesh->mFaces, mesh->mNumFaces);
    std::for_each(shared_futures.begin(), shared_futures.end(), [](std::shared_future<void> &it) -> void { it.wait(); });
    return {i, std::move(mesh_geometry_buffers)};
  };

  /***********************************************************************************************************************************************/
  /* Materials and textures */

  /**
   * The function copies texture data from a GLB file to an ARGB8888 buffer.
   *
   * @param totexture A pointer to a TextureData struct that will hold the copied texture data.
   * @param fromtexture The aiTexture object containing the texture data to be copied.
   */
  static void ax_dbg_optimize0 copyTexels(U32TexData *totexture,
                                          aiTexture *fromtexture,
                                          const std::string &texture_type,
                                          TextureDatabase &texture_database,
                                          std::size_t &texcache_element_count,
                                          controller::IProgressManager &progress_manager) {

    if (fromtexture != nullptr) {
      /* If mHeight != 0 , the texture is uncompressed , and we read it as is */
      if (fromtexture->mHeight != 0) {
        unsigned int width = 0;
        unsigned int height = 0;
        totexture->width = width = fromtexture->mWidth;
        totexture->height = height = fromtexture->mHeight;
        progress_manager.initProgress(std::string("Loading texture  ") + texture_type, static_cast<float>(width * height));
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
        texcache_element_count += 1;
      }
      /* If mHeight = 0 , the texture is compressed , and we need to uncompress and convert it to ARGB32 */
      else
      {
        QImage image;
        std::vector<uint8_t> buffer;
        progress_manager.initProgress(std::string("Loading texture  ") + texture_type, static_cast<float>(fromtexture->mWidth));

        image.loadFromData(reinterpret_cast<const uint8_t *>(fromtexture->pcData), static_cast<int>(fromtexture->mWidth));
        image = image.convertToFormat(QImage::Format_ARGB32);

        unsigned image_width = image.width();
        unsigned image_height = image.height();
        uint32_t *from_buffer = reinterpret_cast<uint32_t *>(image.bits());
        totexture->data = texture_database.copyRangeToCache(from_buffer, nullptr, image_width * image_height, texcache_element_count);
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
                          std::size_t &texcache_element_count,
                          controller::IProgressManager &progress_manager) {
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
      copyTexels(&texture, scene->mTextures[texture_index_int], texture_type, texture_database, texcache_element_count, progress_manager);
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
                                    std::size_t &texcache_element_count,
                                    controller::IProgressManager &progress_manager) {
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
      loadTexture<DiffuseTexture>(scene, &mesh_material, diffuse, color_texture, texture_database, texcache_element_count, progress_manager);
    } else
      loadTextureDummy<DiffuseTexture>(&mesh_material, texture_database);

    if (material->GetTextureCount(aiTextureType_OPACITY) > 0) {
      mesh_material.setAlphaFactor(true);
      material->GetTexture(aiTextureType_OPACITY, 0, &opacity_texture, nullptr, nullptr, nullptr, nullptr, nullptr);
      loadTexture<OpacityTexture>(scene, &mesh_material, opacity, opacity_texture, texture_database, texcache_element_count, progress_manager);
    } else
      loadTextureDummy<OpacityTexture>(&mesh_material, texture_database);

    if (material->GetTextureCount(aiTextureType_METALNESS) > 0) {
      material->GetTexture(AI_MATKEY_METALLIC_TEXTURE, &metallic_texture);
      loadTexture<MetallicTexture>(scene, &mesh_material, metallic, metallic_texture, texture_database, texcache_element_count, progress_manager);
    } else
      loadTextureDummy<MetallicTexture>(&mesh_material, texture_database);

    if (material->GetTextureCount(aiTextureType_DIFFUSE_ROUGHNESS) > 0) {
      material->GetTexture(AI_MATKEY_ROUGHNESS_TEXTURE, &roughness_texture);
      loadTexture<RoughnessTexture>(scene, &mesh_material, roughness, roughness_texture, texture_database, texcache_element_count, progress_manager);
    } else
      loadTextureDummy<RoughnessTexture>(&mesh_material, texture_database);

    if (material->GetTextureCount(aiTextureType_NORMALS) > 0) {
      material->GetTexture(aiTextureType_NORMALS, 0, &normal_texture, nullptr, nullptr, nullptr, nullptr, nullptr);
      loadTexture<NormalTexture>(scene, &mesh_material, normal, normal_texture, texture_database, texcache_element_count, progress_manager);
    } else
      loadTextureDummy<NormalTexture>(&mesh_material, texture_database);

    if (material->GetTextureCount(aiTextureType_LIGHTMAP) > 0) {
      material->GetTexture(aiTextureType_LIGHTMAP, 0, &occlusion_texture, nullptr, nullptr, nullptr, nullptr, nullptr);
      loadTexture<AmbiantOcclusionTexture>(
          scene, &mesh_material, ambiantocclusion, occlusion_texture, texture_database, texcache_element_count, progress_manager);
    } else
      loadTextureDummy<AmbiantOcclusionTexture>(&mesh_material, texture_database);

    if (material->GetTextureCount(aiTextureType_SHEEN) > 0) {
      material->GetTexture(aiTextureType_SHEEN, 0, &specular_texture, nullptr, nullptr, nullptr, nullptr, nullptr);
      loadTexture<SpecularTexture>(scene, &mesh_material, specular, specular_texture, texture_database, texcache_element_count, progress_manager);
    } else
      loadTextureDummy<SpecularTexture>(&mesh_material, texture_database);

    if (material->GetTextureCount(aiTextureType_EMISSIVE) > 0) {
      material->GetTexture(aiTextureType_EMISSIVE, 0, &emissive_texture, nullptr, nullptr, nullptr, nullptr, nullptr);
      loadTexture<EmissiveTexture>(scene, &mesh_material, emissive, emissive_texture, texture_database, texcache_element_count, progress_manager);
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
  /* id will be incremented by the texture caching system*/
  std::pair<unsigned, GLMaterial> loadMaterials(const aiScene *scene,
                                                const aiMaterial *material,
                                                TextureDatabase &texture_database,
                                                std::size_t &texcache_element_count,
                                                controller::IProgressManager &progress_manager) {
    GLMaterial mesh_material = loadAllTextures(scene, material, texture_database, texcache_element_count, progress_manager);
    float transparency_factor = loadTransparencyValue(material);
    mesh_material.setAlphaFactor(transparency_factor);
    return {texcache_element_count, mesh_material};
  }

  static std::size_t total_textures_size(const aiScene *scene) {
    std::size_t totalSize = 0;
    for (unsigned int i = 0; i < scene->mNumTextures; ++i) {
      aiTexture *texture = scene->mTextures[i];
      if (texture->mHeight == 0) {
        QImage image;
        image.loadFromData(reinterpret_cast<uint8_t *>(texture->pcData), static_cast<int>(texture->mWidth));
        image = image.convertToFormat(QImage::Format_ARGB32);
        totalSize += image.width() * image.height() * sizeof(uint32_t);
      } else {
        totalSize += texture->mWidth * texture->mHeight * sizeof(uint32_t);
      }
    }
    return totalSize;
  }

  static std::pair<std::vector<Mesh *>, SceneTree> loadSceneElements(const aiScene *modelScene,
                                                                     ShaderDatabase *shader_database,
                                                                     TextureDatabase *texture_database,
                                                                     INodeDatabase *node_database,
                                                                     controller::IProgressManager &progress_manager) {
    std::vector<std::future<std::pair<unsigned, std::unique_ptr<Object3D>>>> loaded_meshes_futures;
    std::vector<GLMaterial> material_array;
    std::vector<Mesh *> node_lookup_table;
    Shader *shader_program = shader_database->get(Shader::BRDF);
    node_lookup_table.resize(modelScene->mNumMeshes);
    material_array.resize(modelScene->mNumMeshes);
    std::pair<std::vector<Mesh *>, SceneTree> objects;
    std::size_t texture_cache_size = total_textures_size(modelScene);
    texture_database->reserveCache(texture_cache_size, core::memory::PLATFORM_ALIGN);
    std::size_t texcache_element_count = 0;
    for (unsigned int i = 0; i < modelScene->mNumMeshes; i++) {
      loaded_meshes_futures.push_back(std::async(std::launch::async,
                                                 retrieve_geometry,
                                                 modelScene,
                                                 i)  //*We launch multiple threads loading the geometry , and the main thread loads the materials
      );
      aiMaterial *ai_mat = modelScene->mMaterials[modelScene->mMeshes[i]->mMaterialIndex];
      material_array[i] = loadMaterials(modelScene, ai_mat, *texture_database, texcache_element_count, progress_manager).second;
    }
    for (auto &loaded_meshes_future : loaded_meshes_futures) {
      std::pair<unsigned, std::unique_ptr<Object3D>> geometry_loaded = loaded_meshes_future.get();
      unsigned mesh_index = geometry_loaded.first;
      const aiMesh *mesh = modelScene->mMeshes[mesh_index];
      const char *mesh_name = mesh->mName.C_Str();
      std::string name(mesh_name);
      auto mesh_result = database::node::store<Mesh>(
          *node_database, false, name, std::move(*geometry_loaded.second), material_array[mesh_index], shader_program, nullptr);
      LOG("object loaded : " + name, LogLevel::INFO);
      node_lookup_table[mesh_index] = mesh_result.object;
      objects.first.push_back(mesh_result.object);
    }
    SceneTree scene_tree = generateSceneTree(modelScene, node_lookup_table);
    objects.second = scene_tree;
    return objects;
  }

  std::pair<std::vector<Mesh *>, SceneTree> Loader::loadObjects(const char *file) {
    ShaderDatabase *shader_database = resource_database->getShaderDatabase();
    INodeDatabase *node_database = resource_database->getNodeDatabase();
    TextureDatabase *texture_database = resource_database->getTextureDatabase();

    if (!texture_database) {
      LOG("Texture database is not initialized.", LogLevel::CRITICAL);
      return std::pair<std::vector<Mesh *>, SceneTree>();
    }
    if (!node_database) {
      LOG("Node database is not initialized.", LogLevel::CRITICAL);
      return std::pair<std::vector<Mesh *>, SceneTree>();
    }
    if (!shader_database) {
      LOG("Shader database is not initialized.", LogLevel::CRITICAL);
      return std::pair<std::vector<Mesh *>, SceneTree>();
    }

    Assimp::Importer importer;
    const aiScene *modelScene = importer.ReadFile(
        file, aiProcess_CalcTangentSpace | aiProcess_Triangulate | aiProcess_JoinIdenticalVertices | aiProcess_FlipUVs);
    if (modelScene) {
      return loadSceneElements(modelScene, shader_database, texture_database, node_database, progress_manager);
    } else {
      LOG("Cannot read file.", LogLevel::ERROR);
      return std::pair<std::vector<Mesh *>, SceneTree>();
    }
  }

  /**
   * The function loads a file and generates a vector of meshes, including a cube map if applicable.
   * @param file A pointer to a character array representing the file path of the 3D model to be loaded.
   * @return A vector of Mesh pointers.
   */
  std::pair<std::vector<Mesh *>, SceneTree> Loader::load(const char *file) {
    std::pair<std::vector<Mesh *>, SceneTree> scene = loadObjects(file);
    return scene;
  }

}  // namespace IO
