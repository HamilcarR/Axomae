#ifndef TEXTUREGROUP_H
#define TEXTUREGROUP_H

#include "Texture.h"
#include "TextureDatabase.h"

/**
 * @file TextureGroup.h
 * File defining the TextureGroup structure
 */

/**
 * @class TextureGroup
 * @brief A TextureGroup object packs a group of different textures to be bound by the shader
 */
class TextureGroup final {
 private:
  TextureDatabase *texture_database{};   /**<Pointer to the database texture*/
  std::vector<int> texture_collection{}; /**<Array of Pointers to textures in the texture database*/
  bool initialized{};                    /**<State of the textures*/

 public:
  TextureGroup();
  ~TextureGroup() = default;
  TextureGroup(TextureGroup &&move) noexcept;
  TextureGroup(const TextureGroup &copy);
  TextureGroup &operator=(TextureGroup &&move) noexcept;
  TextureGroup &operator=(const TextureGroup &copy);
  virtual void addTexture(int texture_database_index);
  virtual void initializeGlTextureData(Shader *shader);
  virtual void clean();
  virtual void bind();
  virtual void unbind();
  [[nodiscard]] bool isInitialized() const { return initialized; };
  bool containsType(Texture::TYPE type);
  [[nodiscard]] Texture *getTexturePointer(Texture::TYPE type) const;
  [[nodiscard]] const std::vector<int> &getTextureCollection() const { return texture_collection; }
  [[nodiscard]] bool isEmpty() const { return texture_collection.empty(); }
  /**
   * @brief This method will check if every ID references a valid texture in the database.
   * If not , the ID is removed .
   */
  void synchronizeWithDatabaseState();
  bool removeTexture(Texture::TYPE type);
  bool removeTexture(int id);
};

#endif
