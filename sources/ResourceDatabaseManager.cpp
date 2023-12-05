
#include "../includes/ResourceDatabaseManager.h"

std::unique_ptr<ResourceDatabaseManager> ResourceDatabaseManager::instance = nullptr;

ResourceDatabaseManager &ResourceDatabaseManager::getInstance() {
  if (instance == nullptr)
    instance = std::unique_ptr<ResourceDatabaseManager>(new ResourceDatabaseManager());
  return *instance.get();
}

void ResourceDatabaseManager::destroyInstance() {
  instance = nullptr;
}

void ResourceDatabaseManager::purge() {
  if (texture_database)
    texture_database->purge();
  if (shader_database)
    shader_database->purge();
}

ResourceDatabaseManager::ResourceDatabaseManager() {
  texture_database = std::make_unique<TextureDatabase>();
  shader_database = std::make_unique<ShaderDatabase>();
}

ResourceDatabaseManager::~ResourceDatabaseManager() {
  shader_database = nullptr;
  texture_database = nullptr;
}

void ResourceDatabaseManager::cleanShaderDatabase() const {
  shader_database->clean();
}

void ResourceDatabaseManager::cleanTextureDatabase() const {
  texture_database->clean();
}

void ResourceDatabaseManager::purgeShaderDatabase() const {
  shader_database->purge();
}

void ResourceDatabaseManager::purgeTextureDatabase() const {
  texture_database->purge();
}