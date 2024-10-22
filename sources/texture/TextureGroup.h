#ifndef TEXTUREGROUP_H
#define TEXTUREGROUP_H

#include "Texture.h"

/**
 * @file TextureGroup.h
 * File defining the TextureGroup structure
 */

class TextureDatabase;
/**
 * @class TextureGroup
 * @brief A TextureGroup object packs a group of different textures to be bound by the shader
 */
class TextureGroup {
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
  ax_no_discard bool isInitialized() const { return initialized; };
  bool containsType(GenericTexture::TYPE type);
  ax_no_discard GenericTexture *getTexturePointer(GenericTexture::TYPE type) const;
  ax_no_discard const std::vector<int> &getTextureCollection() const { return texture_collection; }
  ax_no_discard bool isEmpty() const { return texture_collection.empty(); }
  /**
   * @brief This method will check if every ID references a valid texture in the database.
   * If not , the ID is removed .
   */
  void synchronizeWithDatabaseState();
  bool removeTexture(GenericTexture::TYPE type);
  bool removeTexture(int id);
};

#endif
