#ifndef LOADER_H
#define LOADER_H

#include "Image.h"
#include "Mesh.h"
#include "ResourceDatabaseManager.h"
#include "SceneHierarchy.h"
#include "constants.h"

#include <boost/core/span.hpp>

/**
 * @file Loader.h
 * Implements a Loader class that will read mesh data and textures from disk
 *
 */

namespace IO {
  struct loader_data_t {
    boost::span<uint8_t> texture_cache;
    std::vector<Mesh *> mesh_list;
    SceneTree scene_tree;
  };

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
