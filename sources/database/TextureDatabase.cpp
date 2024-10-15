#include "TextureDatabase.h"
#include "internal/memory/MemoryArena.h"
#include "internal/thread/Mutex.h"
#include <utility>
TextureDatabase::TextureDatabase(core::memory::ByteArena *arena, controller::ProgressStatus *progress_manager_) {
  progress_manager = progress_manager_;
  setUpCacheMemory(arena);
}

void TextureDatabase::purge() {
  Mutex::Lock lock(mutex);
  for (std::pair<const int, database::Storage<int, GenericTexture>> &A : database_map) {
    A.second.get()->clean();
    A.second.setValidity(false);
  }
  database_map.clear();
  unique_textures.clear();
}
struct DestroyList {
  int id;
  std::string name;
};
void TextureDatabase::clean() {
  std::vector<DestroyList> to_destroy;
  Mutex::Lock lock(mutex);
  for (auto &it : database_map)
    if (!it.second.isPersistent()) {
      it.second.get()->clean();
      it.second.setValidity(false);
      DestroyList dest = {it.first, it.second.get()->getName()};
      to_destroy.push_back(dest);
    }
  for (const auto &elem : to_destroy) {
    unique_textures.erase(elem.name);
    database_map.erase(elem.id);
  }
}

std::vector<database::Result<int, GenericTexture>> TextureDatabase::getTexturesByType(GenericTexture::TYPE type) const {
  std::vector<database::Result<int, GenericTexture>> type_collection;
  Mutex::Lock lock(mutex);
  for (const auto &A : database_map) {
    if (A.second.get()->getTextureType() == type) {
      database::Result<int, GenericTexture> result = {A.first, A.second.get()};
      type_collection.push_back(result);
    }
  }
  return type_collection;
}

database::Result<int, GenericTexture> TextureDatabase::add(std::unique_ptr<GenericTexture> texture, bool keep) {
  bool dummy = texture->isDummyTexture();
  GenericTexture::TYPE type = texture->getTextureType();
  GenericTexture *ptr = texture.get();
  std::string name = texture->getName();
  /* Checks if the texture is named and exists in the database (to avoid duplicates)*/
  if (!name.empty()) {
    auto it = unique_textures.find(name);
    if (it != unique_textures.end()) {
      int id = it->second;
      ptr = get(id);
      return {id, ptr};
    }
  }
  /* If the texture is not named , we store it in a new slot :
   * If it is not a dummy , looks for the first available slot */
  if (!dummy) {
    int id = firstFreeId();
    database::Storage<int, GenericTexture> storage(std::move(texture), id, keep);
    Mutex::Lock lock(mutex);
    database_map[id] = std::move(storage);
    unique_textures.insert(std::pair<std::string, int>(name, id));
    return {id, ptr};
  }
  /* If the texture is not named , and is a dummy , looks for a slot containing a dummy of the same type
   * Returns it if found . Else , looks for the first available slot
   */
  else
  {
    for (auto &elem : database_map) {
      if (elem.second.get()->getTextureType() == type && elem.second.get()->isDummyTexture())
        return {elem.first, elem.second.get()};
    }
    int id = firstFreeId();
    Mutex::Lock lock(mutex);
    database::Storage<int, GenericTexture> storage(std::move(texture), id, true);  // Persistence for dummies is always true.
    database_map[id] = std::move(storage);
    unique_textures.insert(std::pair<std::string, int>(name, id));
    return {id, ptr};
  }
}

bool TextureDatabase::remove(const int index) {
  GenericTexture *tex = get(index);
  Mutex::Lock lock(mutex);
  if (tex) {
    tex->clean();
    database_map.erase(index);
    removeUniqueTextureReference(index);
    return true;
  }
  return false;
}

bool TextureDatabase::remove(const GenericTexture *address) {
  Mutex::Lock lock(mutex);
  for (auto it = database_map.begin(); it != database_map.end(); it++) {
    if (address && it->second.get() == address) {
      it->second.get()->clean();
      database_map.erase(it);
      removeUniqueTextureReference(it->first);
      return true;
    }
  }
  return false;
}

database::Result<int, GenericTexture> TextureDatabase::getUniqueTexture(const std::string &name) const {
  auto it = unique_textures.find(name);
  if (it != unique_textures.end()) {
    return {it->second, get(it->second)};
  }
  return {0, nullptr};
}
bool TextureDatabase::removeUniqueTextureReference(int id) {
  for (auto &elem : unique_textures)
    if (elem.second == id) {
      unique_textures.erase(elem.first);
      return true;
    }
  return false;
}
