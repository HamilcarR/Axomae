#ifndef TEXTUREDATABASE_H
#define TEXTUREDATABASE_H

#include "Axomae_macros.h"
#include "Factory.h"
#include "RenderingDatabaseInterface.h"
#include "Texture.h"

/**
 * @file TextureDatabase.h
 *
 */

/**
 * @brief TextureDatabase class definition
 * GL texture database with no duplicates.
 *
 */
class TextureDatabase final : public IntegerResourceDB<Texture> {
 private:
  std::map<std::string, int> unique_textures;

 public:
  explicit TextureDatabase(controller::ProgressStatus *progress_manager = nullptr);
  ~TextureDatabase() override = default;
  void clean() override;
  void purge() override;
  bool remove(int index) override;
  bool remove(const Texture *texture) override;
  /**
   * @brief Add a texture object to the database . In case the object is already present , this method will return the
   * existing texture's id
   * @param keep True if texture is to be kept
   * @return int Database ID of the texture
   */
  database::Result<int, Texture> add(std::unique_ptr<Texture> texture, bool keep) override;
  std::vector<database::Result<int, Texture>> getTexturesByType(Texture::TYPE texture_type) const;
  bool empty() const override { return database_map.empty(); }
  database::Result<int, Texture> getUniqueTexture(const std::string &name) const;
  bool removeUniqueTextureReference(int id);
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
