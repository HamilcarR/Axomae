#include "../includes/TextureDatabase.h"
#include "../includes/Mutex.h"
#include <algorithm>
#include <utility>

void TextureDatabase::purge() {
  Mutex::Lock lock(mutex);
  for (std::pair<const int, std::unique_ptr<Texture>> &A : database) {
    A.second->clean();
    A.second = nullptr;
  }
  database.clear();
}

void TextureDatabase::clean() {
  std::vector<std::map<const int, std::unique_ptr<Texture>>::iterator> to_destroy;
  Mutex::Lock lock(mutex);
  for (auto it = database.begin(); it != database.end(); it++)
    if (it->first >= 0) {
      it->second->clean();
      it->second = nullptr;
      to_destroy.push_back(it);
    }
  for (auto it : to_destroy)
    database.erase(it);
}

TextureDatabase::TextureDatabase() {}

Texture *TextureDatabase::get(const int index) const {
  Mutex::Lock lock(mutex);
  auto it = database.find(index);
  return it == database.end() ? nullptr : it->second.get();
}

bool TextureDatabase::contains(const int index) const {
  Mutex::Lock lock(mutex);
  return database.find(index) != database.end();
}

std::vector<database::Result<int, Texture>> TextureDatabase::getTexturesByType(Texture::TYPE type) const {
  std::vector<database::Result<int, Texture>> type_collection;
  Mutex::Lock lock(mutex);
  for (const auto &A : database) {
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
    while (database[index] != nullptr) {
      if (database[index]->isDummyTexture() && database[index]->getTextureType() == texture->getTextureType())
        return {index, database[index].get()};
      index--;
    }
    database[index] = std::move(texture);
    return {index, database[index].get()};
  } else {
    int index = 0;
    while (database[index] != nullptr)
      index++;
    database[index] = std::move(texture);
    return {index, database[index].get()};
  }
}

database::Result<int, Texture> TextureDatabase::contains(const Texture *address) const {
  if(!address)
    return {0 , nullptr}; 
  Mutex::Lock lock(mutex);
  for (const auto &A : database)
    if (A.second.get() == address)
      return {A.first, A.second.get()};
  return {0, nullptr};
}

bool TextureDatabase::remove(const int index) {
  Texture *tex = get(index);
  if (tex) {
    tex->clean();
    database.erase(index);
    return true;
  }
  return false;
}

bool TextureDatabase::remove(const Texture *address) {
  for (auto it = database.begin(); it != database.end(); it++) {
    if (address && it->second.get() == address) {
      it->second->clean();
      database.erase(it);
      return true;
    }
  }
  return false;
}