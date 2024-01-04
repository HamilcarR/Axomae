#include "TextureDatabase.h"
#include "Mutex.h"
#include <utility>

void TextureDatabase::purge() {
  Mutex::Lock lock(mutex);
  for (std::pair<const int, database::Storage<int, Texture>> &A : database_map) {
    A.second.get()->clean();
    A.second.setValidity(false);
  }
  database_map.clear();
  unique_textures.clear();
}
struct DestroyList{
    int id ;
    std::string name;
};
void TextureDatabase::clean() {
  std::vector<DestroyList> to_destroy;
  Mutex::Lock lock(mutex);
  for (auto & it : database_map)
    if (!it.second.isPersistent()) {
      it.second.get()->clean();
      it.second.setValidity(false);
      to_destroy.push_back({it.first , it.second.get()->getName()});
    }
  for (const auto& elem : to_destroy) {
    unique_textures.erase(elem.name);
    database_map.erase(elem.id);
  }
}

std::vector<database::Result<int, Texture>> TextureDatabase::getTexturesByType(Texture::TYPE type) const {
  std::vector<database::Result<int, Texture>> type_collection;
  Mutex::Lock lock(mutex);
  for (const auto &A : database_map) {
    if (A.second.get()->getTextureType() == type) {
      database::Result<int, Texture> result = {A.first, A.second.get()};
      type_collection.push_back(result);
    }
  }
  return type_collection;
}

/* will add real textures by pushing them into the first key available.
 * Dummy textures , on the other hand will only occupy one key , and the key will be returned if we try to add
 * one of the same type . This is for avoiding duplicates for dummies .
 * Persistence for dummies is always true .
 */

database::Result<int, Texture> TextureDatabase::add(std::unique_ptr<Texture> texture, bool keep) {
  bool dummy = texture->isDummyTexture();
  Texture::TYPE type = texture->getTextureType();
  Texture *ptr = texture.get();
  std::string name = texture->getName();
  /* Checks if the texture is present in the database by name*/
  if (!name.empty()) {
    auto it = unique_textures.find(name);
    if (it != unique_textures.end()) {
      int id = it->second;
      ptr = get(id);
      return {id, ptr};
    }
  }
  /* Case for non dummy textures : create one when there's no texture with the same name , and save name + id in "unique_textures" map*/
  if (!dummy) {
    int id = firstFreeId();
    database::Storage<int, Texture> storage(std::move(texture), id, keep);
    Mutex::Lock lock(mutex);
    database_map[id] = std::move(storage);
    unique_textures.insert(std::pair<std::string, int>(name, id));
    return {id, ptr};
  } else {
    int id = 0;
    for (auto &elem : database_map) {
      if (elem.second.get()->getTextureType() == type && elem.second.get()->isDummyTexture())
        return {elem.first, elem.second.get()};
      id++;
    }
    Mutex::Lock lock(mutex);
    database::Storage<int, Texture> storage(std::move(texture), id, true);  // always keep dummies
    database_map[id] = std::move(storage);
    unique_textures.insert(std::pair<std::string, int>(name, id));
    return {id, ptr};
  }
}

bool TextureDatabase::remove(const int index) {
  Texture *tex = get(index);
  Mutex::Lock lock(mutex);
  if (tex) {
    tex->clean();
    database_map.erase(index);
    return true;
  }
  return false;
}

bool TextureDatabase::remove(const Texture *address) {
  Mutex::Lock lock(mutex);
  for (auto it = database_map.begin(); it != database_map.end(); it++) {
    if (address && it->second.get() == address) {
      it->second.get()->clean();
      database_map.erase(it);
      return true;
    }
  }
  return false;
}

/*Replace invalid data with new values*/
int TextureDatabase::firstFreeId() const {
  int diff = 0;
  for (const auto &elem : database_map) {
    if (!elem.second.isValid())
      return elem.first;
    if ((elem.first - diff) > 1)
      return (elem.first + diff) / 2;
    diff = elem.first;
  }
  return size();
}

database::Result<int, Texture> TextureDatabase::getUniqueTexture(const std::string &name) const {
  auto it = unique_textures.find(name);
  if (it != unique_textures.end()) {
    return {it->second, get(it->second)};
  }
  return {0, nullptr};
}