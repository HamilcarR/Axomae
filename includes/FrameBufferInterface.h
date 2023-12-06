#ifndef FRAMEBUFFERINTERFACE_H
#define FRAMEBUFFERINTERFACE_H

#include "Drawable.h"
#include "GLFrameBuffer.h"
#include "Mesh.h"
#include "ShaderDatabase.h"
#include "Texture.h"

/**
 * @file FrameBufferInterface.h
 * This file implements an interface for framebuffers
 *
 */

/**
 * @class FrameBufferInterface
 * This class implements a generic framebuffer
 */
class FrameBufferInterface {
 public:
  /**
   * @brief Construct a new Frame Buffer Interface object
   *
   */
  FrameBufferInterface();

  /**
   * @brief Construct a new Frame Buffer Interface object
   *
   * @param texture_database
   * @param texture_size
   * @param default_fbo_id_pointer
   * @param rendertype
   */
  FrameBufferInterface(TextureDatabase *texture_database,
                       ScreenSize *texture_size,
                       unsigned int *default_fbo_id_pointer = nullptr);

  /**
   * @brief Destroy the Frame Buffer Interface object
   *
   */
  virtual ~FrameBufferInterface();

  /**
   * @brief Resizes the textures used by the framebuffer .
   * Will use the values stored inside the texture_dim pointer property
   *
   */
  virtual void resize();

  /**
   * @brief Set new screen dimensions
   * @param pointer_on_texture_size Pointer on texture size
   *
   */
  virtual void setTextureDimensions(ScreenSize *pointer_on_texture_size) {
    texture_dim = pointer_on_texture_size;
  }

  /**
   * @brief Render to the texture stored inside the framebuffer
   *
   */
  virtual void bindFrameBuffer();

  /**
   * @brief Unbind the framebuffer , and use the default framebuffer.
   *
   */
  virtual void unbindFrameBuffer();

  /**
   * @brief Initializes shaders , and render buffers textures .
   *
   */
  virtual void initializeFrameBuffer();

  /**
   * @brief Clean the framebuffer's data
   *
   */
  virtual void clean();

  /**
   * @brief Set the Default Frame Buffer Id Pointer object
   *
   * @param id
   */
  virtual void setDefaultFrameBufferIdPointer(unsigned *id) {
    default_framebuffer_pointer = id;
  }

  /**
   * @brief Get the texture used for a color attachment.
   *
   * @param color_attachment
   * @return Texture*
   */
  Texture *getFrameBufferTexturePointer(GLFrameBuffer::INTERNAL_FORMAT color_attachment) {
    return fbo_attachment_texture_collection[color_attachment];
  }

  /**
   * @brief Initialize an empty target texture to be rendered to , saves it in the database , and returns it's database
   * ID
   *
   * @param database The texture database.
   * @param width Width of the target texture
   * @param height Height of the target texture
   * @param persistence If true , keeps the texture after texture database cleanup
   * @param internal_format Internal format of the texture to generate
   * @param data_format Data format of the texture
   * @param data_type Type of the data for the texture
   * @param type Type of the target texture , can be of type Texture::FRAMEBUFFER , or Texture::CUBEMAP
   * @param mipmaps Level of mipmaps for this texture
   * @return int Database ID of this texture
   */
  template<class TEXTYPE>
  int setUpEmptyTexture(unsigned width,
                        unsigned height,
                        bool persistence,
                        Texture::FORMAT internal_format,
                        Texture::FORMAT data_format,
                        Texture::FORMAT data_type,
                        unsigned int mipmaps = 0) {
    TextureData temp_empty_data_texture;
    temp_empty_data_texture.width = width;
    temp_empty_data_texture.height = height;
    temp_empty_data_texture.internal_format = internal_format;
    temp_empty_data_texture.data_format = data_format;
    temp_empty_data_texture.data_type = data_type;
    temp_empty_data_texture.mipmaps = mipmaps;
    return texture_database->addTexture<TEXTYPE>(&temp_empty_data_texture, persistence);
  }
  /**
   * @brief Calls setUpEmptyTexture() , with these args , and store the resulting texture to be rendered into in the
   * database
   *
   * @param color_attachment
   * @param persistence
   * @param internal_format
   * @param data_format
   * @param data_type
   * @param width
   * @param height
   * @param rendertype
   * @param mipmaps
   */

  template<class TEXTYPE>
  void initializeFrameBufferTexture(GLFrameBuffer::INTERNAL_FORMAT color_attachment,
                                    bool persistence,
                                    Texture::FORMAT internal_format,
                                    Texture::FORMAT data_format,
                                    Texture::FORMAT data_type,
                                    unsigned width,
                                    unsigned height,
                                    unsigned int mipmaps = 0) {

    unsigned int texture_id = setUpEmptyTexture<TEXTYPE>(
        width, height, persistence, internal_format, data_format, data_type, mipmaps);
    fbo_attachment_texture_collection[color_attachment] = texture_database->get(texture_id);
  }

 protected:
  GLFrameBuffer *gl_framebuffer_object;
  ScreenSize *texture_dim;
  TextureDatabase *texture_database;
  std::map<GLFrameBuffer::INTERNAL_FORMAT, Texture *> fbo_attachment_texture_collection;
  unsigned int *default_framebuffer_pointer;  //! use as argument in constructor , or getters and setters
};

#endif