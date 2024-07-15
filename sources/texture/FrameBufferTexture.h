#ifndef FRAMEBUFFERTEXTURE_H
#define FRAMEBUFFERTEXTURE_H
#include "GenericTexture.h"

class FrameBufferTexture : public GenericTexture {
 protected:
  FrameBufferTexture();
  /**
   * @brief Construct a new Frame Buffer Texture
   * Contains only width , and height... The rest of the TextureData parameter
   * is not used,
   * @param data TextureData parameter
   *
   */
  explicit FrameBufferTexture(TextureData *data);
  FrameBufferTexture(unsigned width, unsigned height);

 public:
  void bind() override;
  void unbind() override;
  void initialize(Shader *shader) override;
  void initializeTexture2D() override;
  static const char *getTextureTypeCStr();
  [[nodiscard]] TYPE getTextureType() const override { return FRAMEBUFFER; }
};

#endif  // FRAMEBUFFERTEXTURE_H
