#include "TextureDatabase.h"
#include "Mutex.h"
#include <utility>


void TextureDatabase::purge() {
  Mutex::Lock lock(mutex);
  for (std::pair<const int, std::unique_ptr<Texture>> &A : database_map) {
    A.second->clean();
    A.second = nullptr;
  }
  database_map.clear();
}

void TextureDatabase::clean() {
  std::vector<std::map<int, std::unique_ptr<Texture>>::iterator> to_destroy;
  Mutex::Lock lock(mutex);
  for (auto it = database_map.begin(); it != database_map.end(); it++)
    if (it->first >= 0) {
      it->second->clean();
      it->second = nullptr;
      to_destroy.push_back(it);
    }
  for (auto it : to_destroy)
    database_map.erase(it);
}

std::vector<database::Result<int, Texture>> TextureDatabase::getTexturesByType(Texture::TYPE type) const {
  std::vector<database::Result<int, Texture>> type_collection;
  Mutex::Lock lock(mutex);
  for (const auto &A : database_map) {
    if (A.second->getTextureType() == type) {
      database::Result<int, Texture> result = {A.first, A.second.get()};
      type_collection.push_back(result);
    }
  }
  return type_collection;
}

database::Result<int, Texture> TextureDatabase::add(std::unique_ptr<Texture> texture, bool keep) {
  Mutex::Lock lock(mutex);
  if (keep) {
    int index = -1;
    while (database_map[index] != nullptr) {
      if (database_map[index]->isDummyTexture() && database_map[index]->getTextureType() == texture->getTextureType())
        return {index, database_map[index].get()};
      index--;
    }
    database_map[index] = std::move(texture);
    return {index, database_map[index].get()};
  } else {
    int index = 0;
    while (database_map[index] != nullptr)
      index++;
    database_map[index] = std::move(texture);
    return {index, database_map[index].get()};
  }
}

bool TextureDatabase::remove(const int index) {
  Texture *tex = get(index);
  if (tex) {
    tex->clean();
    database_map.erase(index);
    return true;
  }
  return false;
}

bool TextureDatabase::remove(const Texture *address) {
  for (auto it = database_map.begin(); it != database_map.end(); it++) {
    if (address && it->second.get() == address) {
      it->second->clean();
      database_map.erase(it);
      return true;
    }
  }
  return false;
}