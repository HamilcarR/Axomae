#ifndef TEXTUREFACTORY_H
#define TEXTUREFACTORY_H

#include "Axomae_macros.h"
#include "Factory.h"
#include "RenderingDatabaseInterface.h"
#include "Texture.h"
/**
 * @file TextureFactory.h
 * Class definition of the factory system for the textures
 *
 */

/**
 * @class TextureFactory
 * @brief Provides a way to easily create textures from raw data
 *
 */
class TextureBuilder {

 public:
  /**
   * @brief Construct a new Texture* object from raw texture data , and a type
   *
   * @param data Contains raw data about the texture
   * @return std::unique_texture<Texture> The created texture object
   * @see Texture
   */
  template<class TEXTYPE, class... Args>
  static std::unique_ptr<TEXTYPE> build(Args &&...args) {
    ASSERT_SUBTYPE(Texture, TEXTYPE);
    return std::make_unique<PRVINTERFACE<TEXTYPE, Args...>>(std::forward<Args>(args)...);
  }

  template<class TEXTYPE, class... Args>
  static factory::Result<int, TEXTYPE> store(IResourceDB<int, Texture> &database, bool keep, Args &&...args) {
    ASSERT_SUBTYPE(Texture, TEXTYPE);
    std::unique_ptr<Texture> temp = std::make_unique<PRVINTERFACE<TEXTYPE, Args...>>(std::forward<Args>(args)...);
    TEXTYPE *pointer = static_cast<TEXTYPE *>(temp.get());
    factory::Result<int, TEXTYPE> result = {database.add(std::move(temp), keep), pointer};
    return result;
  }
};

#endif
