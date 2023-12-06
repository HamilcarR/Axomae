#ifndef TEXTUREFACTORY_H
#define TEXTUREFACTORY_H

#include "Axomae_macros.h"
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
  template<class TEXTYPE>
  static std::unique_ptr<TEXTYPE> build(TextureData *data) {
    ASSERT_SUBTYPE(Texture, TEXTYPE);
    class PRVINTERFACE : public TEXTYPE {
     public:
      PRVINTERFACE(TextureData *data) : TEXTYPE(data) {}
    };
    return std::make_unique<PRVINTERFACE>(data);
  }
};

#endif
