#include "INodeDatabase.h"
#include "INodeFactory.h"
#include "Loader.h"
#include "ResourceDatabaseManager.h"
#include "ShaderDatabase.h"
#include "ShaderFactory.h"
#include "TextureDatabase.h"
#include "TextureFactory.h"
#include "string/axomae_str_utils.h"
#include <QImage>

#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>

// TODO : code needs total uncoupling from any foreign structure : needs to return arrays of basic data

namespace IO {
  namespace exception {}

  Loader::Loader(controller::ProgressStatus *prog_stat) : resource_database(&ResourceDatabaseManager::getInstance()) {
    setProgressManager(prog_stat);
  }

  /**
   * The function copies texture data from a GLB file to an ARGB8888 buffer.
   *
   * @param totexture A pointer to a TextureData struct that will hold the copied texture data.
   * @param fromtexture The aiTexture object containing the texture data to be copied.
   */
  static void copyTexels(TextureData *totexture,
                         aiTexture *fromtexture,
                         const std::string &texture_type,
                         controller::IProgressManager *progress_manager) {

    if (fromtexture != nullptr) {
      /* If mHeight != 0 , the texture is uncompressed , and we read it as is */
      if (fromtexture->mHeight != 0) {
        unsigned int width = 0;
        unsigned int height = 0;
        totexture->width = width = fromtexture->mWidth;
        totexture->height = height = fromtexture->mHeight;
        totexture->data.resize(totexture->width * totexture->height);

        progress_manager->initProgress(std::string("Loading texture  ") + texture_type, static_cast<float>(width * height));
        controller::ProgressManagerHelper progress_helper(progress_manager);

        for (unsigned int i = 0; i < width * height; i++) {
          progress_helper.notifyProgress(i);
          uint8_t a = fromtexture->pcData[i].a;
          uint8_t r = fromtexture->pcData[i].r;
          uint8_t g = fromtexture->pcData[i].g;
          uint8_t b = fromtexture->pcData[i].b;
          uint32_t rgba = (a << 24) | (b << 16) | (g << 8) | r;
          totexture->data[i] = rgba;
        }
      }
      /* If mHeight = 0 , the texture is compressed , and we need to uncompress and convert it to ARGB32 */
      else
      {
        QImage image;
        std::vector<uint8_t> buffer;
        progress_manager->initProgress(std::string("Loading texture  ") + texture_type, static_cast<float>(fromtexture->mWidth));
        controller::ProgressManagerHelper progress_helper(progress_manager);
        progress_helper.notifyProgress(controller::ProgressManagerHelper::ONE_FOURTH);

        image.loadFromData((const unsigned char *)fromtexture->pcData, static_cast<int>(fromtexture->mWidth));
        image = image.convertToFormat(QImage::Format_ARGB32);
        progress_helper.notifyProgress(controller::ProgressManagerHelper::THREE_FOURTH);

        unsigned image_width = image.width();
        unsigned image_height = image.height();
        totexture->data.resize(image_width * image_height);
        uint8_t *dest_buffer = reinterpret_cast<uint8_t *>(&totexture->data[0]);
        uint8_t *from_buffer = image.bits();
        std::memcpy(dest_buffer, from_buffer, image_height * image_width * sizeof(uint32_t));
        progress_helper.notifyProgress(controller::ProgressManagerHelper::COMPLETE);
        totexture->width = image_width;
        totexture->height = image_height;
        progress_helper.reset();
        LOG("image of size " + std::to_string(totexture->width) + " x " + std::to_string(totexture->height) + " uncompressed ", LogLevel::INFO);
      }
    }
  }

  template<class TEXTYPE>
  static void loadTextureDummy(GLMaterial *material, TextureDatabase *texture_database) {
    auto result = database::texture::store<TEXTYPE>(*texture_database, true, nullptr);
    LOG("Loading dummy texture at index : " + std::to_string(result.id), LogLevel::INFO);
    material->addTexture(result.id);
  }

  /**
   * This function loads a texture from an aiScene and adds it to a Material object.
   *
   * @param scene A pointer to the aiScene object which contains the loaded 3D model data.
   * @param material A pointer to a Material object that the texture will be added to.
   * @param texture The variable that stores the loaded texture data.
   * @param texture_string The name or index of the texture file to be loaded, stored as an aiString.
   * @param type The type of texture being loaded, which is of the enum type Texture::TYPE.
   */
  template<class TEXTYPE>
  static void loadTexture(const aiScene *scene,
                          GLMaterial *material,
                          TextureData *texture,
                          const aiString &texture_string,
                          ResourceDatabaseManager &resource_manager,
                          controller::IProgressManager *progress_manager) {
    std::string texture_index_string = texture_string.C_Str();
    std::string texture_type = texture->name;
    TextureDatabase &texture_database = *resource_manager.getTextureDatabase();
    RawImageDatabase &image_database = *resource_manager.getRawImgdatabase();
    if (!texture_index_string.empty()) {
      texture_index_string = texture_index_string.substr(1);  // get rid of the '*' character at the beginning of the string id
      texture->name = texture_index_string;
      auto result = texture_database.getUniqueTexture(texture->name);  // Check if the name (the assimp texture id number) is present in the database.
      if (result.object != nullptr) {
        material->addTexture(result.id);
      } else {
        int texture_index_int = stoi(texture_index_string);                                        // convert id to integer
        copyTexels(texture, scene->mTextures[texture_index_int], texture_type, progress_manager);  // read the image pixels and copy them to "texture"
        auto result_add = database::texture::store<TEXTYPE>(texture_database, false, texture);     // add new ID
        image::Metadata metadata;
        metadata.width = texture->width;
        metadata.height = texture->height;
        metadata.channels = 4;
        metadata.is_hdr = false;
        metadata.name = texture->name;

        std::vector<uint8_t> bytearray(metadata.width * metadata.height * metadata.channels);
        std::memcpy(bytearray.data(), texture->data.data(), metadata.width * metadata.height * metadata.channels);
        material->addTexture(result_add.id);
      }
    } else
      LOG("Loader can't load texture\n", LogLevel::WARNING);
  }

  /**
   * The function loads textures for a given material in a 3D model scene.
   *
   * @param scene a pointer to the aiScene object which contains the loaded 3D model data.
   * @param material The aiMaterial object that contains information about the material properties of a
   * 3D model.
   *
   * @return a Material object.
   */
  static GLMaterial loadAllTextures(const aiScene *scene,
                                    const aiMaterial *material,
                                    ResourceDatabaseManager &resource_manager,
                                    controller::IProgressManager *progress_manager) {
    GLMaterial mesh_material;
    std::vector<GenericTexture::TYPE> dummy_textures_type;
    TextureData diffuse, metallic, roughness, normal, ambiantocclusion, emissive, specular, opacity;
    diffuse.name = "diffuse";
    metallic.name = "metallic";
    roughness.name = "roughness";
    opacity.name = "opacity";
    normal.name = "normal";
    ambiantocclusion.name = "occlusion";
    specular.name = "specular";
    emissive.name = "emissive";
    unsigned int color_index = 0, metallic_index = 0, roughness_index = 0;
    aiString color_texture, opacity_texture, normal_texture, metallic_texture, roughness_texture, emissive_texture, specular_texture,
        occlusion_texture;  // we get indexes of embedded textures , since we will use GLB format
    if (material->GetTextureCount(aiTextureType_BASE_COLOR) > 0) {
      material->GetTexture(AI_MATKEY_BASE_COLOR_TEXTURE, &color_texture);
      loadTexture<DiffuseTexture>(scene, &mesh_material, &diffuse, color_texture, resource_manager, progress_manager);
    } else
      loadTextureDummy<DiffuseTexture>(&mesh_material, resource_manager.getTextureDatabase());

    if (material->GetTextureCount(aiTextureType_OPACITY) > 0) {
      mesh_material.setAlphaFactor(true);
      material->GetTexture(aiTextureType_OPACITY, 0, &opacity_texture, nullptr, nullptr, nullptr, nullptr, nullptr);
      loadTexture<OpacityTexture>(scene, &mesh_material, &opacity, opacity_texture, resource_manager, progress_manager);
    } else
      loadTextureDummy<OpacityTexture>(&mesh_material, resource_manager.getTextureDatabase());

    if (material->GetTextureCount(aiTextureType_METALNESS) > 0) {
      material->GetTexture(AI_MATKEY_METALLIC_TEXTURE, &metallic_texture);
      loadTexture<MetallicTexture>(scene, &mesh_material, &metallic, metallic_texture, resource_manager, progress_manager);
    } else
      loadTextureDummy<MetallicTexture>(&mesh_material, resource_manager.getTextureDatabase());

    if (material->GetTextureCount(aiTextureType_DIFFUSE_ROUGHNESS) > 0) {
      material->GetTexture(AI_MATKEY_ROUGHNESS_TEXTURE, &roughness_texture);
      loadTexture<RoughnessTexture>(scene, &mesh_material, &roughness, roughness_texture, resource_manager, progress_manager);
    } else
      loadTextureDummy<RoughnessTexture>(&mesh_material, resource_manager.getTextureDatabase());

    if (material->GetTextureCount(aiTextureType_NORMALS) > 0) {
      material->GetTexture(aiTextureType_NORMALS, 0, &normal_texture, nullptr, nullptr, nullptr, nullptr, nullptr);
      loadTexture<NormalTexture>(scene, &mesh_material, &normal, normal_texture, resource_manager, progress_manager);
    } else
      loadTextureDummy<NormalTexture>(&mesh_material, resource_manager.getTextureDatabase());

    if (material->GetTextureCount(aiTextureType_LIGHTMAP) > 0) {
      material->GetTexture(aiTextureType_LIGHTMAP, 0, &occlusion_texture, nullptr, nullptr, nullptr, nullptr, nullptr);
      loadTexture<AmbiantOcclusionTexture>(scene, &mesh_material, &ambiantocclusion, occlusion_texture, resource_manager, progress_manager);
    } else
      loadTextureDummy<AmbiantOcclusionTexture>(&mesh_material, resource_manager.getTextureDatabase());

    if (material->GetTextureCount(aiTextureType_SHEEN) > 0) {
      material->GetTexture(aiTextureType_SHEEN, 0, &specular_texture, nullptr, nullptr, nullptr, nullptr, nullptr);
      loadTexture<SpecularTexture>(scene, &mesh_material, &specular, specular_texture, resource_manager, progress_manager);
    } else
      loadTextureDummy<SpecularTexture>(&mesh_material, resource_manager.getTextureDatabase());

    if (material->GetTextureCount(aiTextureType_EMISSIVE) > 0) {
      material->GetTexture(aiTextureType_EMISSIVE, 0, &emissive_texture, nullptr, nullptr, nullptr, nullptr, nullptr);
      loadTexture<EmissiveTexture>(scene, &mesh_material, &emissive, emissive_texture, resource_manager, progress_manager);
    } else
      loadTextureDummy<EmissiveTexture>(&mesh_material, resource_manager.getTextureDatabase());

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

  std::pair<unsigned, GLMaterial> loadMaterials(const aiScene *scene,
                                                const aiMaterial *material,
                                                ResourceDatabaseManager &resource_manager,
                                                unsigned id,
                                                controller::IProgressManager *progress_manager) {
    GLMaterial mesh_material = loadAllTextures(scene, material, resource_manager, progress_manager);
    float transparency_factor = loadTransparencyValue(material);
    mesh_material.setAlphaFactor(transparency_factor);
    return {id, mesh_material};
  }

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
    assert(node != nullptr);
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
   * The function "geometry_fill_buffers" fills the buffers of a 3D object with vertex, normal, tangent,
   * bitangent, and UV data from an imported model.
   *
   * @param modelScene modelScene is a pointer to an aiScene object, which represents a 3D model scene.
   * It contains information about the model's meshes, materials, textures, and other properties.
   * @param i The parameter "i" is the index of the mesh in the model scene that you want to fill the
   * buffers for.
   *
   * @return a std::pair<unsigned, Object3D*>.
   */
  std::pair<unsigned, std::unique_ptr<Object3D>> geometry_fill_buffers(const aiScene *modelScene, unsigned i) {
    const aiMesh *mesh = modelScene->mMeshes[i];
    auto object = std::make_unique<Object3D>();
    auto size_dim3 = mesh->mNumVertices * 3;
    auto size_dim2 = mesh->mNumVertices * 2;
    auto size_dim3_indices = mesh->mNumFaces * 3;
    object->vertices.resize(size_dim3);
    object->normals.resize(size_dim3);
    object->tangents.resize(size_dim3);
    object->bitangents.resize(size_dim3);
    object->indices.resize(size_dim3_indices);
    object->uv.resize(size_dim2);
    assert(mesh->HasTextureCoords(0));
    std::future<void> f_vertices, f_normals, f_bitangents, f_tangents, f_uv;
    f_vertices = std::async(std::launch::async, [&]() { load_geometry_buffer(object->vertices, mesh->mVertices, mesh->mNumVertices, 3); });
    f_normals = std::async(std::launch::async, [&]() { load_geometry_buffer(object->normals, mesh->mNormals, mesh->mNumVertices, 3); });
    f_bitangents = std::async(std::launch::async, [&]() { load_geometry_buffer(object->bitangents, mesh->mBitangents, mesh->mNumVertices, 3); });
    f_tangents = std::async(std::launch::async, [&]() { load_geometry_buffer(object->tangents, mesh->mTangents, mesh->mNumVertices, 3); });
    f_uv = std::async(std::launch::async, [&]() { load_geometry_buffer(object->uv, mesh->mTextureCoords[0], mesh->mNumVertices, 2); });
    load_indices_buffer(object->indices, mesh->mFaces, mesh->mNumFaces);
    std::vector<std::shared_future<void>> shared_futures = {
        f_vertices.share(), f_normals.share(), f_bitangents.share(), f_tangents.share(), f_uv.share()};
    std::for_each(shared_futures.begin(), shared_futures.end(), [](std::shared_future<void> &it) -> void { it.wait(); });
    return {i, std::move(object)};
  };

  /**
   * The function "loadObjects" loads objects from a file using the Assimp library and returns a pair
   * containing a vector of meshes and a scene tree.
   *
   * @param file The "file" parameter is a const char pointer that represents the file path of the scene
   * file that needs to be loaded.
   *
   * @return a pair containing a vector of Mesh pointers and a SceneTree object.
   */

  std::pair<std::vector<Mesh *>, SceneTree> Loader::loadObjects(const char *file) {
    ShaderDatabase &shader_database = *resource_database->getShaderDatabase();
    INodeDatabase &node_database = *resource_database->getNodeDatabase();

    std::pair<std::vector<Mesh *>, SceneTree> objects;
    Assimp::Importer importer;
    const aiScene *modelScene = importer.ReadFile(
        file, aiProcess_CalcTangentSpace | aiProcess_Triangulate | aiProcess_JoinIdenticalVertices | aiProcess_FlipUVs);
    if (modelScene != nullptr) {
      std::vector<std::future<std::pair<unsigned, std::unique_ptr<Object3D>>>> loaded_meshes_futures;
      std::vector<GLMaterial> material_array;
      std::vector<Mesh *> node_lookup_table;
      Shader *shader_program = shader_database.get(Shader::BRDF);
      node_lookup_table.resize(modelScene->mNumMeshes);
      material_array.resize(modelScene->mNumMeshes);
      for (unsigned int i = 0; i < modelScene->mNumMeshes; i++) {
        loaded_meshes_futures.push_back(std::async(std::launch::async,
                                                   geometry_fill_buffers,
                                                   modelScene,
                                                   i)  //*We launch multiple threads loading the geometry , and the main thread loads the materials
        );
        aiMaterial *ai_mat = modelScene->mMaterials[modelScene->mMeshes[i]->mMaterialIndex];
        material_array[i] = loadMaterials(modelScene, ai_mat, *resource_database, i, this).second;
      }
      for (auto it = loaded_meshes_futures.begin(); it != loaded_meshes_futures.end(); it++) {
        std::pair<unsigned, std::unique_ptr<Object3D>> geometry_loaded = it->get();
        unsigned mesh_index = geometry_loaded.first;
        const aiMesh *mesh = modelScene->mMeshes[mesh_index];
        const char *mesh_name = mesh->mName.C_Str();
        std::string name(mesh_name);
        auto mesh_result = database::node::store<Mesh>(
            node_database, false, name, std::move(*geometry_loaded.second), material_array[mesh_index], shader_program, nullptr);
        LOG("object loaded : " + name, LogLevel::INFO);
        node_lookup_table[mesh_index] = mesh_result.object;
        objects.first.push_back(mesh_result.object);
      }
      SceneTree scene_tree = generateSceneTree(modelScene, node_lookup_table);
      objects.second = scene_tree;
      return objects;
    } else {
      LOG("Problem loading scene", LogLevel::ERROR);
      return std::pair<std::vector<Mesh *>, SceneTree>();
    }
  }

  /**
   * The function loads a file and generates a vector of meshes, including a cube map if applicable.
   *
   * @param file A pointer to a character array representing the file path of the 3D model to be loaded.
   *
   * @return A vector of Mesh pointers.
   */
  std::pair<std::vector<Mesh *>, SceneTree> Loader::load(const char *file) {
    TextureDatabase *texture_database = resource_database->getTextureDatabase();
    texture_database->clean();
    std::pair<std::vector<Mesh *>, SceneTree> scene = loadObjects(file);
    return scene;
  }

}  // namespace IO
