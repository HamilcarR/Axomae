#ifndef TEXTUREDATABASE_H
#define TEXTUREDATABASE_H

#include "Axomae_macros.h"
#include "Factory.h"
#include "RenderingDatabaseInterface.h"
#include "Texture.h"

/**
 * @file TextureDatabase.h
 * Class definition for the texture database
 *
 */

/**
 * @brief TextureDatabase class definition
 * The TextureDatabase class is a singleton that holds an std::map with unique key associated with a texture type
 * We use it to keep texture objects in one place , meshes will only reference the textures.
 *
 */
class TextureDatabase final : public IResourceDB<int, Texture> {
 public:
  /**
   * @brief Construct a new Texture Database object
   *
   */
  TextureDatabase() = default;
  /**
   * @brief Removes all elements from the database , except those marked "keep"
   *
   */
  void clean() override;

  /**
   * @brief Deletes everything in the database
   *
   */
  void purge() override;

  /**
   * @brief Removes a texture from the database using it's ID
   *
   * @param index
   * @return true
   * @return false
   */
   bool remove(int index) override;

  /**
   * @brief Removes a texture from the database using it's address
   *
   * @param texture Texture to remove
   * @return true If the texture has been found
   * @return false If the address is not in the database
   */
  bool remove(const Texture *texture) override;

  /**
   * @brief Add a texture object to the database . In case the object is already present , this method will return the
   * already present texture's id
   *
   * @param texture Texture object to add
   * @param keep True if texture is to be kept
   * @return int Database ID of the texture
   */
  virtual database::Result<int, Texture> add(std::unique_ptr<Texture> texture, bool keep);

  /**
   * @brief Retrieve all textures of type "texture_type"
   *
   * @param texture_type Type of the texture
   * @return std::vector<std::pair<unsigned int , Texture*>> List of all textures matching "texture_type"
   * @see Texture
   * @see Texture::TYPE
   */
  std::vector<database::Result<int, Texture>> getTexturesByType(Texture::TYPE texture_type) const;
  bool empty() const override { return database_map.empty(); }
  const std::map<int, std::unique_ptr<Texture>> &getConstData() const override { return database_map; }

 private:
};

namespace database::texture {

  template<class TEXTYPE, class... Args>
  static database::Result<int, TEXTYPE> store(IResourceDB<int, Texture> &database, bool keep, Args &&...args) {
    ASSERT_SUBTYPE(Texture, TEXTYPE);
    std::unique_ptr<Texture> temp = std::make_unique<PRVINTERFACE<TEXTYPE, Args...>>(std::forward<Args>(args)...);
    database::Result<int, Texture> result = database.add(std::move(temp), keep);
    database::Result<int, TEXTYPE> cast = {result.id, static_cast<TEXTYPE *>(result.object)};
    return cast;
  }
};  // namespace database::texture

#endif
