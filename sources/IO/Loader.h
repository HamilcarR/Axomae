#ifndef LOADER_H
#define LOADER_H

#include <cassert>
#include <cstdlib>
#include <iostream>
#include <memory>

#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>

#include "EnvmapController.h"
#include "GenericException.h"
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
  namespace exception {
    class LoadImagePathException : public GenericException {
     public:
      explicit LoadImagePathException(const std::string &path) { GenericException::saveErrorString(std::string("Failed processing path : ") + path); }
    };

    class LoadImageDimException : public GenericException {
     public:
      explicit LoadImageDimException(int width, int height) {
        std::string dim = std::string("width : ") + std::to_string(width) + std::string(" height:") + std::to_string(height);
        GenericException::saveErrorString(std::string("Image dimensions error: ") + dim);
      }
    };

    class LoadImageChannelException : public GenericException {
     public:
      explicit LoadImageChannelException(int channels) {
        std::string chan = std::string("channels number : ") + std::to_string(channels);
        GenericException::saveErrorString(std::string("Image channel error: ") + chan);
      }
    };
  }  // namespace exception

  /**
   * @brief 3D Loader class
   */
  class Loader : public controller::IProgressManager {
   public:
    /**
     * @brief Construct a new Loader object
     *
     */
    explicit Loader(controller::ProgressStatus *progress_status);

    /**
     * @brief Load a .glb file
     *
     * @param file Path of the 3D glb model
     * @return std::vector<Mesh*>
     */
    std::pair<std::vector<Mesh *>, SceneTree> load(const char *file);

    /**
     * @brief Loads a text file , with new lines
     *
     * @param filename Path of the file
     * @return std::string
     */
    std::string loadTextFile(const char *filename);

    /**
     * @brief Loads all meshes from the GLB file.
     * @param filename GLB file path
     * @return std::vector<Mesh*>
     */
    std::pair<std::vector<Mesh *>, SceneTree> loadObjects(const char *filename);

    /**
     * @brief Loads an HDR image .
     * @param store if true , stores the loaded data in the Image db .
     * @return Image array
     */
    image::ImageHolder<float> loadHdr(const char *path, bool store = true);

    void writeHdr(const char *path, const image::ImageHolder<float> &image);

   private:
    ResourceDatabaseManager *resource_database;
  };

}  // namespace IO

#endif
