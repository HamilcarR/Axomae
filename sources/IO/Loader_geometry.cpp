#include "INodeDatabase.h"
#include "INodeFactory.h"
#include "Loader.h"
#include "ResourceDatabaseManager.h"
#include "ShaderDatabase.h"
#include <assimp/scene.h>

/***********************************************************************************************************************************************/
/* Geometry */

namespace IO {
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

  static SceneTreeNode *fillTreeData(aiNode *ai_node, const std::vector<Mesh *> &mesh_lookup, SceneTreeNode *parent) {
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


  template<class T>
  static void load_geometry_buffer(axstd::span<T> &dest, const aiVector3D *from, int size, int dimension) {
    for (int f = 0, i = 0; f < size; f++) {
      const aiVector3D &vect = from[f];
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

  static void load_indices_buffer(axstd::span<unsigned> &dest, const aiFace *faces, int num_faces) {
    for (int i = 0, f = 0; i < num_faces; i++, f += 3) {
      assert(faces[i].mNumIndices == 3);
      dest[f] = faces[i].mIndices[0];
      dest[f + 1] = faces[i].mIndices[1];
      dest[f + 2] = faces[i].mIndices[2];
    }
  }

  std::size_t total_geometry_size(const aiScene *scene, controller::IProgressManager *progress_manager) {
    std::size_t total_size = 0;
    progress_manager->initProgress("Computing geometry cache size ", static_cast<float>(scene->mNumMeshes));
    for (std::size_t mesh_index = 0; mesh_index < scene->mNumMeshes; mesh_index++) {
      const aiMesh *mesh = scene->mMeshes[mesh_index];
      /* vertices , normals , bitangents , tangents */
      unsigned geo_buffer_present = 1, uv_buffer_present = 0;
      if (mesh->HasNormals())
        geo_buffer_present++;
      if (mesh->HasTangentsAndBitangents())
        geo_buffer_present += 2;
      const std::size_t float_buffers_3d = geo_buffer_present * 3 * sizeof(float);
      /* uv */
      if (mesh->HasTextureCoords(0))
        uv_buffer_present++;
      const std::size_t float_buffers_2d = uv_buffer_present * 2 * sizeof(float);
      /* indices */
      const std::size_t uint_buffers_3d = 3 * sizeof(unsigned int);

      total_size += mesh->mNumVertices * float_buffers_3d;
      total_size += mesh->mNumVertices * float_buffers_2d;
      total_size += mesh->mNumFaces * uint_buffers_3d;
      progress_manager->setCurrent((float)mesh_index);
      progress_manager->notifyProgress();
    }
    progress_manager->resetProgress();
    return total_size;
  }

  template<class T>
  static axstd::span<T> retrieve_buffer_and_update_offset(const INodeDatabase& mesh_database, std::size_t &offset , unsigned attrib_count) {
    uint8_t* vertex_attributes_cache = mesh_database.getCurrentCache();
    AX_ASSERT_NEQ(vertex_attributes_cache, nullptr);
    axstd::span<T> buffer = axstd::span<T>(reinterpret_cast<T*>(vertex_attributes_cache + offset), attrib_count);
    offset += attrib_count * sizeof(T);
    return buffer;
  }

  /**
   * This function loads vertex attributes in an async manner.
   * element_count in bytes.
   */
  std::pair<unsigned, std::unique_ptr<Object3D>> load_geometry(const aiScene *modelScene,
                                                               unsigned mesh_index,
                                                               INodeDatabase &mesh_database,
                                                               std::size_t &offset) {
    const aiMesh *mesh = modelScene->mMeshes[mesh_index];
    AX_ASSERT_NOTNULL(mesh->mVertices);
    std::unique_ptr<Object3D> mesh_geometry_buffers = std::make_unique<Object3D>();
    auto dim3_attrib_count = mesh->mNumVertices * 3;
    auto dim2_attrib_count = mesh->mNumVertices * 2;
    auto dim3_faces_count = mesh->mNumFaces * 3;

    std::vector<std::shared_future<void>> shared_futures;
    std::future<void> f_vertices, f_normals, f_bitangents, f_tangents, f_uv;
    mesh_geometry_buffers->vertices = retrieve_buffer_and_update_offset<float>(mesh_database , offset, dim3_attrib_count);
    f_vertices = std::async(std::launch::async,
                            [&]() { load_geometry_buffer(mesh_geometry_buffers->vertices, mesh->mVertices, mesh->mNumVertices, 3); });
    shared_futures.push_back(f_vertices.share());

    if (mesh->HasNormals()) {
      mesh_geometry_buffers->normals = retrieve_buffer_and_update_offset<float>(mesh_database , offset, dim3_attrib_count);
      f_normals = std::async(std::launch::async,
                             [&]() { load_geometry_buffer(mesh_geometry_buffers->normals, mesh->mNormals, mesh->mNumVertices, 3); });
      shared_futures.push_back(f_normals.share());
    }

    if (mesh->HasTextureCoords(0)) {
      mesh_geometry_buffers->uv = retrieve_buffer_and_update_offset<float>(mesh_database , offset, dim2_attrib_count);
      f_uv = std::async(std::launch::async,
                        [&]() { load_geometry_buffer(mesh_geometry_buffers->uv, mesh->mTextureCoords[0], mesh->mNumVertices, 2); });
      shared_futures.push_back(f_uv.share());
    }

    if (mesh->HasTangentsAndBitangents()) {
      mesh_geometry_buffers->tangents = retrieve_buffer_and_update_offset<float>(mesh_database , offset, dim3_attrib_count);
      mesh_geometry_buffers->bitangents = retrieve_buffer_and_update_offset<float>(mesh_database , offset, dim3_attrib_count);
      f_tangents = std::async(std::launch::async,
                              [&]() { load_geometry_buffer(mesh_geometry_buffers->tangents, mesh->mTangents, mesh->mNumVertices, 3); });
      f_bitangents = std::async(std::launch::async,
                                [&]() { load_geometry_buffer(mesh_geometry_buffers->bitangents, mesh->mBitangents, mesh->mNumVertices, 3); });
      shared_futures.push_back(f_tangents.share());
      shared_futures.push_back(f_bitangents.share());
    }
    mesh_geometry_buffers->indices = retrieve_buffer_and_update_offset<unsigned>(mesh_database , offset, dim3_faces_count);
    load_indices_buffer(mesh_geometry_buffers->indices, mesh->mFaces, mesh->mNumFaces);
    std::for_each(shared_futures.begin(), shared_futures.end(), [](std::shared_future<void> &it) -> void { it.wait(); });
    return {mesh_index, std::move(mesh_geometry_buffers)};
  }
}  // namespace IO