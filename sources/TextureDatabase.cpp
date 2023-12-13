#include "../includes/TextureDatabase.h"
#include "../includes/Mutex.h"
#include <algorithm>
#include <utility>

void TextureDatabase::purge() {
  Mutex::Lock lock(mutex);
  for (std::pair<const int, std::unique_ptr<Texture>> &A : texture_database) {
    A.second->clean();
    A.second = nullptr;
  }
  texture_database.clear();
}

void TextureDatabase::clean() {
  std::vector<std::map<const int, std::unique_ptr<Texture>>::iterator> to_destroy;
  Mutex::Lock lock(mutex);
  for (auto it = texture_database.begin(); it != texture_database.end(); it++)
    if (it->first >= 0) {
      it->second->clean();
      it->second = nullptr;
      to_destroy.push_back(it);
    }
  for (auto it : to_destroy)
    texture_database.erase(it);
}

TextureDatabase::TextureDatabase() {}

Texture *TextureDatabase::get(const int index) const {
  Mutex::Lock lock(mutex);
  auto it = texture_database.find(index);
  return it == texture_database.end() ? nullptr : it->second.get();
}

bool TextureDatabase::contains(const int index) const {
  Mutex::Lock lock(mutex);
  return texture_database.find(index) != texture_database.end();
}

std::vector<database::Result<int, Texture>> TextureDatabase::getTexturesByType(Texture::TYPE type) const {
  std::vector<database::Result<int, Texture>> type_collection;
  Mutex::Lock lock(mutex);
  for (const auto &A : texture_database) {
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
    while (texture_database[index] != nullptr) {
      if (texture_database[index]->isDummyTexture() && texture_database[index]->getTextureType() == texture->getTextureType())
        return {index, texture_database[index].get()};
      index--;
    }
    texture_database[index] = std::move(texture);
    return {index, texture_database[index].get()};
  } else {
    int index = 0;
    while (texture_database[index] != nullptr)
      index++;
    texture_database[index] = std::move(texture);
    return {index, texture_database[index].get()};
  }
}

database::Result<int, Texture> TextureDatabase::contains(const Texture *address) const {
  Mutex::Lock lock(mutex);
  for (const auto &A : texture_database)
    if (A.second.get() == address)
      return {A.first, A.second.get()};
  return {0, nullptr};
}

bool TextureDatabase::remove(const int index) {
  Texture *tex = get(index);
  if (tex) {
    tex->clean();
    texture_database.erase(index);
    return true;
  }
  return false;
}

bool TextureDatabase::remove(const Texture *address) {
  for (auto it = texture_database.begin(); it != texture_database.end(); it++) {
    if (address && it->second.get() == address) {
      it->second->clean();
      texture_database.erase(it);
      return true;
    }
  }
  return false;
}