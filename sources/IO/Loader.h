#ifndef LOADER_H
#define LOADER_H

#include <cassert>
#include <cstdlib>
#include <iostream>
#include <memory>

#include "GenericException.h"
#include "Image.h"
#include "Mesh.h"
#include "ResourceDatabaseManager.h"
#include "SceneHierarchy.h"
#include "constants.h"

/**
 * @file Loader.h
 * Implements a Loader class that will read mesh data and textures from disk
 *
 */

namespace IO {

  /**
   * @brief 3D Loader class
   */
  class Loader : public controller::IProgressManager {

   private:
    ResourceDatabaseManager *resource_database;

   public:
    explicit Loader(controller::ProgressStatus *progress_status);

    /**
     * @brief Load a .glb file
     * @param file Path of the 3D glb model
     * @return std::vector<Mesh*>
     */
    std::pair<std::vector<Mesh *>, SceneTree> load(const char *file);

    /**
     * @brief Loads a text file , with new lines
     * @param filename Path of the file
     * @return std::string
     */
    std::string loadTextFile(const char *filename);

    /**
     * @brief Loads all meshes from the GLB file.
     */
    std::pair<std::vector<Mesh *>, SceneTree> loadObjects(const char *filename);

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
