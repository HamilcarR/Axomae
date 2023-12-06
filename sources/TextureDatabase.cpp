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
TextureDatabase::~TextureDatabase() {}

Texture *TextureDatabase::get(const int index) {
  Mutex::Lock lock(mutex);
  return texture_database[index].get();
}

bool TextureDatabase::contains(const int index) {
  Mutex::Lock lock(mutex);
  return texture_database.find(index) != texture_database.end();
}

std::vector<std::pair<int, Texture *>> TextureDatabase::getTexturesByType(Texture::TYPE type) {
  std::vector<std::pair<int, Texture *>> type_collection;
  Mutex::Lock lock(mutex);
  for (auto &A : texture_database) {
    if (A.second->getTextureType() == type) {
      type_collection.push_back(std::pair<int, Texture *>(A.first, A.second.get()));
    }
  }
  return type_collection;
}

int TextureDatabase::add(std::unique_ptr<Texture> texture, bool keep) {
  Mutex::Lock lock(mutex);
  if (keep) {
    int index = -1;
    while (texture_database[index] != nullptr)
      index--;
    texture_database[index] = std::move(texture);
    return index;
  } else {
    int index = 0;
    while (texture_database[index] != nullptr)
      index++;
    texture_database[index] = std::move(texture);
    return index;
  }
}

std::pair<int, Texture *> TextureDatabase::contains(const Texture *address) {
  Mutex::Lock lock(mutex);
  for (auto &A : texture_database)
    if (A.second.get() == address)
      return std::pair(A.first, A.second.get());
  return std::pair(0, nullptr);
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