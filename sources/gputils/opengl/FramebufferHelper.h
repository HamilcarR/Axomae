#ifndef FRAMEBUFFERHELPER_H
#define FRAMEBUFFERHELPER_H

#include "FramebufferHelperInterface.h"
#include "TextureDatabase.h"
#include "internal/device/rendering/opengl/GLFrameBuffer.h"

class TextureDatabase;

class FramebufferHelper : public FramebufferHelperInterface {
 protected:
  std::unique_ptr<GLFrameBuffer> gl_framebuffer_object{};
  Dim2 *texture_dim{};
  TextureDatabase *texture_database{};
  std::map<GLFrameBuffer::INTERNAL_FORMAT, GenericTexture *> fbo_attachment_texture_collection{};
  unsigned int *default_framebuffer_pointer{};

 public:
  CLASS_OM(FramebufferHelper)
  FramebufferHelper(TextureDatabase *texture_database, Dim2 *texture_size, unsigned int *default_fbo_id_pointer = nullptr);
  void resize() override;
  void setTextureDimensions(Dim2 *pointer_on_texture_size) override;
  void bind() override;
  void unbind() override;
  void initialize() override;
  void clean() override;
  [[nodiscard]] bool isReady() const override;
  void setDefaultFrameBufferIdPointer(unsigned *id) override;
  GenericTexture *getFrameBufferTexturePointer(GLFrameBuffer::INTERNAL_FORMAT color_attachment);
  [[nodiscard]] GLFrameBuffer *getFramebufferObject() const { return gl_framebuffer_object.get(); }
  /**
   * @brief Initialize an empty target texture to be rendered to , saves it in the database , and returns it's database
   * ID
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
                        GenericTexture::FORMAT internal_format,
                        GenericTexture::FORMAT data_format,
                        GenericTexture::FORMAT data_type,
                        unsigned int mipmaps = 0);

  /**
   * @brief Calls setUpEmptyTexture() , with these args , and store the resulting texture to be rendered into in the
   * database
   */
  template<class TEXTYPE>
  void initializeFrameBufferTexture(GLFrameBuffer::INTERNAL_FORMAT color_attachment,
                                    bool persistence,
                                    GenericTexture::FORMAT internal_format,
                                    GenericTexture::FORMAT data_format,
                                    GenericTexture::FORMAT data_type,
                                    unsigned width,
                                    unsigned height,
                                    unsigned int mipmaps = 0);
};

/************************************************************************************************************************************/

template<class TEXTYPE>
int FramebufferHelper::setUpEmptyTexture(unsigned width,
                                         unsigned height,
                                         bool persistence,
                                         GenericTexture::FORMAT internal_format,
                                         GenericTexture::FORMAT data_format,
                                         GenericTexture::FORMAT data_type,
                                         unsigned int mipmaps) {
  TextureData temp_empty_data_texture;
  temp_empty_data_texture.width = width;
  temp_empty_data_texture.height = height;
  temp_empty_data_texture.internal_format = internal_format;
  temp_empty_data_texture.data_format = data_format;
  temp_empty_data_texture.data_type = data_type;
  temp_empty_data_texture.mipmaps = mipmaps;
  database::Result<int, TEXTYPE> result = database::texture::store<TEXTYPE>(*texture_database, persistence, &temp_empty_data_texture);
  return result.id;
}

/**
 * @brief Calls setUpEmptyTexture() , with these args , and store the resulting texture to be rendered into in the
 * database
 */
template<class TEXTYPE>
void FramebufferHelper::initializeFrameBufferTexture(GLFrameBuffer::INTERNAL_FORMAT color_attachment,
                                                     bool persistence,
                                                     GenericTexture::FORMAT internal_format,
                                                     GenericTexture::FORMAT data_format,
                                                     GenericTexture::FORMAT data_type,
                                                     unsigned width,
                                                     unsigned height,
                                                     unsigned int mipmaps) {

  unsigned int texture_id = setUpEmptyTexture<TEXTYPE>(width, height, persistence, internal_format, data_format, data_type, mipmaps);
  fbo_attachment_texture_collection[color_attachment] = texture_database->get((int)texture_id);
}
#endif