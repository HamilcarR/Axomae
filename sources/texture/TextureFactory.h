#ifndef TEXTUREFACTORY_H
#define TEXTUREFACTORY_H
#include "Texture.h"
#include "internal/common/Factory.h"
#include "internal/macro/project_macros.h"
/**
 * @file TextureFactory.h
 * Class definition of the factory system for the textures
 */

/**
 * @class TextureFactory
 * @brief Provides a way to easily create textures from raw data
 */
class TextureBuilder {

 public:
  /**
   * @brief Construct a new Texture* object from raw texture data , and a type
   * @param data Contains raw data about the texture
   * @return std::unique_texture<Texture> The created texture object
   * @see Texture
   */
  template<class TEXTYPE, class... Args>
  ax_maybe_unused static std::unique_ptr<TEXTYPE> build(Args &&...args) {
    ASSERT_SUBTYPE(GenericTexture, TEXTYPE);
    return std::make_unique<PRVINTERFACE<TEXTYPE, Args...>>(std::forward<Args>(args)...);
  }
};

#endif
