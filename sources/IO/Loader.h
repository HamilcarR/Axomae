#ifndef LOADER_H
#define LOADER_H

#include "Image.h"
#include "Mesh.h"
#include "ResourceDatabaseManager.h"
#include "SceneHierarchy.h"
#include "constants.h"

#include <internal/common/axstd/span.h>

/**
 * @file Loader.h
 * Implements a Loader class that will read mesh data and textures from disk
 *
 */

class aiScene;
namespace IO {
  struct loader_data_t {
    axstd::span<uint8_t> texture_cache;
    axstd::span<uint8_t> geometry_cache;
    std::vector<Mesh *> mesh_list;
    SceneTree scene_tree;
  };

  std::size_t total_textures_size(const aiScene *scene);
  std::size_t total_geometry_size(const aiScene *scene);
  std::pair<unsigned, std::unique_ptr<Object3D>> load_geometry(const aiScene *modelScene,
                                                               unsigned mesh_index,
                                                               INodeDatabase &mesh_database,
                                                               std::size_t &geocache_element_count,
                                                               controller::IProgressManager &progress_manager);

  std::pair<unsigned, GLMaterial> load_materials(const aiScene *scene,
                                                 unsigned mesh_material_index,
                                                 TextureDatabase &texture_database,
                                                 std::size_t &texcache_element_count,
                                                 controller::IProgressManager &progress_manager);

  SceneTree generateSceneTree(const aiScene *modelScene, const std::vector<Mesh *> &node_lookup);

  /**
   * @brief 3D Loader class
   */
  class Loader {

   private:
    ResourceDatabaseManager *resource_database;
    controller::IProgressManager progress_manager;

   public:
    explicit Loader(controller::ProgressStatus *progress_status);

    /**
     * @brief Load a .glb file
     * @param file Path of the 3D glb model
     * @return std::vector<Mesh*>
     */
    loader_data_t load(const char *file);

    /**
     * @brief Loads a text file , with new lines
     * @param filename Path of the file
     * @return std::string
     */
    std::string loadTextFile(const char *filename);

    /**
     * @brief Loads all meshes from the GLB file.
     */
    loader_data_t loadObjects(const char *filename);

    /**
     * @brief Loads an HDR equirectangular envmap . Alpha channel is clipped .
     * @param store if true , stores the loaded data in the Image db .
     * @return Image array
     */
    image::ImageHolder<float> loadHdrEnvmap(const char *path, bool store = true);

    image::ImageHolder<float> loadRadianceFile(const char *path, bool store);

    image::ImageHolder<float> loadExrFile(const char *path, bool store);

    /**
     * @brief Writes an hdr file on disk , adds extension according to the metadata format of the image (exr or hdr)
     */
    void writeHdr(const char *path, const image::ImageHolder<float> &image, bool flip = false);
  };

}  // namespace IO

#endif
