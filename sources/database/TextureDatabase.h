#ifndef TEXTUREDATABASE_H
#define TEXTUREDATABASE_H
#include "RenderingDatabaseInterface.h"
#include "Texture.h"
#include <internal/common/Factory.h>
#include <internal/macro/project_macros.h>

/**
 * @file TextureDatabase.h
 *
 */

namespace axstd {
  template<class T>
  class MemoryArena;
}

/**
 * @brief TextureDatabase class definition
 * GL texture database with no duplicates.
 *
 */
class TextureDatabase final : public IntegerResourceDB<GenericTexture> {
 private:
  std::map<std::string, int> unique_textures;

 public:
  explicit TextureDatabase(axstd::ByteArena *memory_arena = nullptr, controller::ProgressStatus *progress_manager = nullptr);
  ~TextureDatabase() override = default;
  TextureDatabase(const TextureDatabase &other) = delete;
  TextureDatabase(TextureDatabase &&other) noexcept = delete;
  TextureDatabase &operator=(const TextureDatabase &other) = delete;
  TextureDatabase &operator=(TextureDatabase &&other) noexcept = delete;

  void clean() override;
  void purge() override;
  bool remove(int index) override;
  bool remove(const GenericTexture *texture) override;
  /**
   * @brief Add a texture object to the database . In case the object is already present , this method will return the
   * existing texture's id
   * @param keep True if texture is to be kept
   * @return int Database ID of the texture
   */
  database::Result<int, GenericTexture> add(std::unique_ptr<GenericTexture> texture, bool keep) override;
  std::vector<database::Result<int, GenericTexture>> getTexturesByType(GenericTexture::TYPE texture_type) const;
  bool empty() const override { return database_map.empty(); }
  database::Result<int, GenericTexture> getUniqueTexture(const std::string &name) const;
  bool removeUniqueTextureReference(int id);
};

namespace database::texture {
  template<class TEXTYPE, class... Args>
  database::Result<int, TEXTYPE> store(IResourceDB<int, GenericTexture> &database, bool keep, Args &&...args) {
    ASSERT_SUBTYPE(GenericTexture, TEXTYPE);
    std::unique_ptr<GenericTexture> temp = std::make_unique<PRVINTERFACE<TEXTYPE, Args...>>(std::forward<Args>(args)...);
    database::Result<int, GenericTexture> result = database.add(std::move(temp), keep);
    database::Result<int, TEXTYPE> cast = {result.id, static_cast<TEXTYPE *>(result.object)};
    return cast;
  }
};  // namespace database::texture

#endif
