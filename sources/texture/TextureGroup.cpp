#include "TextureGroup.h"
#include "ResourceDatabaseManager.h"
#include "Shader.h"
#include "TextureDatabase.h"

TextureGroup::TextureGroup() : texture_database(ResourceDatabaseManager::getInstance().getTextureDatabase()), initialized(false) {}

TextureGroup::TextureGroup(const TextureGroup &copy) {
  texture_collection = copy.texture_collection;
  initialized = copy.initialized;
  texture_database = copy.texture_database;
}

TextureGroup::TextureGroup(TextureGroup &&move) noexcept {
  texture_collection = std::move(move.texture_collection);
  initialized = move.initialized;
  texture_database = move.texture_database;
}

TextureGroup &TextureGroup::operator=(const TextureGroup &copy) {
  if (this != &copy) {
    texture_collection = copy.getTextureCollection();
    initialized = copy.isInitialized();
    texture_database = copy.texture_database;
  }
  return *this;
}

TextureGroup &TextureGroup::operator=(TextureGroup &&move) noexcept {
  if (this != &move) {
    texture_collection = std::move(move.texture_collection);
    texture_database = move.texture_database;
    initialized = move.initialized;
  }
  return *this;
}

void TextureGroup::addTexture(int index) { texture_collection.push_back(index); }

GenericTexture *TextureGroup::getTexturePointer(GenericTexture::TYPE type) const {
  for (int id : texture_collection) {
    GenericTexture *A = texture_database->get(id);
    if (A && A->getTextureType() == type)
      return A;
  }
  return nullptr;
}

bool TextureGroup::containsType(GenericTexture::TYPE type) {
  for (int id : texture_collection) {
    GenericTexture *A = texture_database->get(id);
    if (A && A->getTextureType() == type)
      return true;
  }
  return false;
}

void TextureGroup::initializeGlTextureData(Shader *shader) {
  for (int id : texture_collection) {
    GenericTexture *A = texture_database->get(id);
    if (A && !A->isInitialized())
      A->initialize(shader);
  }
  initialized = true;
}

void TextureGroup::clean() {
  initialized = false;
  texture_collection.clear();
}

void TextureGroup::synchronizeWithDatabaseState() {
  std::vector<unsigned int> to_delete;
  unsigned delete_index = 0;
  for (int id : texture_collection) {
    GenericTexture *A = texture_database->get(id);
    if (!A)
      to_delete.push_back(delete_index);
    delete_index++;
  }
  for (unsigned i : to_delete)
    texture_collection.erase(texture_collection.begin() + i);
}

void TextureGroup::bind() {
  std::vector<unsigned int> to_delete;
  unsigned delete_index = 0;
  for (int id : texture_collection) {
    GenericTexture *A = texture_database->get(id);
    if (A)
      A->bind();
    else
      to_delete.push_back(delete_index);
    delete_index++;
  }
  for (unsigned i : to_delete)
    texture_collection.erase(texture_collection.begin() + i);
}

void TextureGroup::unbind() {
  for (int id : texture_collection) {
    GenericTexture *A = texture_database->get(id);
    if (A)
      A->unbind();
  }
}

bool TextureGroup::removeTexture(int id) {
  auto it = std::find(texture_collection.begin(), texture_collection.end(), id);
  if (it != texture_collection.end()) {
    texture_collection.erase(it);
    return true;
  }
  return false;
}

bool TextureGroup::removeTexture(GenericTexture::TYPE type) {
  std::vector collection = texture_database->getTexturesByType(type);
  bool val = false;
  for (auto &elem : collection) {
    auto it = std::find(texture_collection.begin(), texture_collection.end(), elem.id);
    if (it != texture_collection.end()) {
      texture_collection.erase(it);
      val = true;
    }
  }
  return val;
}
