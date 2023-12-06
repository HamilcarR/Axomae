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
class TextureGroup {
 public:
  /**
   * @brief Construct a new Texture Group object
   *
   */
  TextureGroup();

  /**
   * @brief Construct a new Texture Group object
   *
   * @param texture_group
   */
  TextureGroup(const TextureGroup &texture_group);

  /**
   * @brief Destroy the Texture Group object
   *
   */
  virtual ~TextureGroup();

  /**
   * @brief Adds a texture to the collection
   *
   * @param texture_database_index Database index of the texture
   * @param type Type of the texture
   */
  virtual void addTexture(int texture_database_index);

  /**
   * @brief Initialize texture related GL functions and sets up corresponding uniforms
   *
   */
  virtual void initializeGlTextureData(Shader *shader);

  /**
   * @brief Sets the "initialized" variable to false .
   * The cleaning of the textures is done by the TextureDatabase class
   * @see TextureDatabase
   *
   */
  virtual void clean();

  /**
   * @brief Binds every texture in the collection
   *
   */
  virtual void bind();

  /**
   * @brief Unbinds every texture in the collection
   *
   */
  virtual void unbind();

  /**
   * @brief Returns the current state of the texture group
   *
   * @return true if the textures are ready to use
   * @return false if the textures are not ready to use
   */
  bool isInitialized() const {
    return initialized;
  };

  /**
   * @brief Checks if a texture of a certain type exists in the TextureGroup's collection
   *
   * @param type Type of the texture
   * @return true If a texture has been found
   */
  bool containsType(Texture::TYPE type);

  /**
   * @brief Get the Texture of type "type"
   *
   * @param type
   * @return Texture*
   */
  Texture *getTexturePointer(Texture::TYPE type);

  /**
   * @brief Get the Texture Collection object
   *
   * @return std::vector<int>
   */
  const std::vector<int> &getTextureCollection() const {
    return texture_collection;
  }

  /**
   * @brief
   *
   * @param texture_group
   * @return TextureGroup&
   */
  TextureGroup &operator=(const TextureGroup &texture_group);

  /**
   * @brief This method will check if every ID references a valid texture in the database.
   * If not , the ID is removed .
   */
  void synchronizeWithDatabaseState();

 public:
  TextureDatabase *texture_database; /**<Pointer to the database texture*/
 private:
  std::vector<int> texture_collection; /**<Array of Pointers to textures in the texture database*/
  bool initialized;                    /**<State of the textures*/
};

#endif
