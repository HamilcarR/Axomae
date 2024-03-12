
#include "ResourceDatabaseManager.h"
#include "INodeDatabase.h"
#include "ImageDatabase.h"
#include "OP_ProgressStatus.h"
#include "ShaderDatabase.h"
#include "TextureDatabase.h"

ResourceDatabaseManager &ResourceDatabaseManager::getInstance() {
  static ResourceDatabaseManager instance;
  return instance;
}

void ResourceDatabaseManager::purge() {
  texture_database->purge();
  shader_database->purge();
  node_database->purge();
  hdr_database->purge();
  image_database->purge();
}
void ResourceDatabaseManager::clean() {
  texture_database->clean();
  shader_database->clean();
  node_database->clean();
  hdr_database->clean();
  image_database->clean();
}

ResourceDatabaseManager::ResourceDatabaseManager() {}

void ResourceDatabaseManager::initializeDatabases(controller::ProgressStatus *progress_manager) {
  texture_database = std::make_unique<TextureDatabase>(progress_manager);
  shader_database = std::make_unique<ShaderDatabase>(progress_manager);
  node_database = std::make_unique<INodeDatabase>(progress_manager);
  hdr_database = std::make_unique<ImageDatabase<float>>(progress_manager);
  image_database = std::make_unique<ImageDatabase<uint8_t>>(progress_manager);
}

void ResourceDatabaseManager::setProgressManagerAllDb(controller::ProgressStatus *progress_manager) {
  texture_database->setProgressManager(progress_manager);
  shader_database->setProgressManager(progress_manager);
  node_database->setProgressManager(progress_manager);
  hdr_database->setProgressManager(progress_manager);
  image_database->setProgressManager(progress_manager);
}

void ResourceDatabaseManager::cleanShaderDatabase() { shader_database->clean(); }
void ResourceDatabaseManager::purgeShaderDatabase() { shader_database->purge(); }
void ResourceDatabaseManager::cleanTextureDatabase() { texture_database->clean(); }
void ResourceDatabaseManager::purgeTextureDatabase() { texture_database->purge(); }
void ResourceDatabaseManager::cleanNodeDatabase() { node_database->clean(); }
void ResourceDatabaseManager::purgeNodeDatabase() { node_database->purge(); }
void ResourceDatabaseManager::cleanHdrDatabase() { hdr_database->clean(); }
void ResourceDatabaseManager::purgeHdrDatabase() { hdr_database->purge(); }
void ResourceDatabaseManager::cleanRawImgDatabase() { image_database->clean(); }
void ResourceDatabaseManager::purgeRawImgDatabase() { image_database->purge(); }