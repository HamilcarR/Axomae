#ifndef RESOURCEDATABASEMANAGER_H
#define RESOURCEDATABASEMANAGER_H

#include <cstdint>
#include <internal/macro/project_macros.h>
#include <memory>

/**
 * @file ResourceDatabaseManager.h
 * @brief This file implements a singleton containing multiple resources database
 *
 */

class TextureDatabase;
class ShaderDatabase;
class INodeDatabase;

template<class T>
class ImageDatabase;

using RawImageDatabase = ImageDatabase<uint8_t>;

using HdrImageDatabase = ImageDatabase<float>;

namespace controller {
  class OperatorProgressStatus;
  using ProgressStatus = OperatorProgressStatus;
}  // namespace controller

namespace core::memory {
  template<class T>
  class MemoryArena;
}

/**
 * @brief This class contains all databases relative to resources used by the renderer.
 * Every resource loaded using IO is managed from this class.
 *
 */
class ResourceDatabaseManager {
 private:
  std::unique_ptr<TextureDatabase> texture_database; /*<Pointer on the texture database*/
  std::unique_ptr<ShaderDatabase> shader_database;   /*<Pointer on the shader database*/
  std::unique_ptr<INodeDatabase> node_database;      /*<Pointer on the node database*/
  std::unique_ptr<HdrImageDatabase> hdr_database;    /*<Raw HDR images database*/
  std::unique_ptr<RawImageDatabase> image_database;  /*<Raw texture images database*/

 private:
  ResourceDatabaseManager() = default;
  ~ResourceDatabaseManager() = default;

 public:
  static ResourceDatabaseManager &getInstance();
  ResourceDatabaseManager(const ResourceDatabaseManager &) = delete;
  ResourceDatabaseManager &operator=(const ResourceDatabaseManager &) = delete;
  ResourceDatabaseManager &operator=(ResourceDatabaseManager &&) = delete;
  ResourceDatabaseManager(ResourceDatabaseManager &&) = delete;
  /**
   * @brief This method purges everything, deleting every resource stored , and the resources space taken GPU side
   * Additionally , will delete the singleton instance pointer , as well as the databases pointers.
   */
  void purge();
  void clean();
  void cleanTextureDatabase();
  void cleanHdrDatabase();
  void cleanShaderDatabase();
  void cleanNodeDatabase();
  void purgeTextureDatabase();
  void purgeHdrDatabase();
  void purgeShaderDatabase();
  void purgeNodeDatabase();
  void cleanRawImgDatabase();
  void purgeRawImgDatabase();
  ax_no_discard TextureDatabase *getTextureDatabase() const { return texture_database.get(); }
  ax_no_discard ShaderDatabase *getShaderDatabase() const { return shader_database.get(); }
  ax_no_discard INodeDatabase *getNodeDatabase() const { return node_database.get(); }
  ax_no_discard HdrImageDatabase *getHdrDatabase() const { return hdr_database.get(); }
  ax_no_discard RawImageDatabase *getRawImgdatabase() const { return image_database.get(); }
  void setProgressManagerAllDb(controller::ProgressStatus *progress_manager);

  void initializeDatabases(core::memory::MemoryArena<std::byte> &arena, controller::ProgressStatus *progress_manager = nullptr);
};

#endif