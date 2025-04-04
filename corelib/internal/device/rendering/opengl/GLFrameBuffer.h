#ifndef GLFRAMEBUFFER_H
#define GLFRAMEBUFFER_H

#include "../DeviceBufferInterface.h"
#include "GLRenderBuffer.h"
#include <memory>
class GLFrameBuffer : public DeviceMutableBufferInterface {
 public:
  enum INTERNAL_FORMAT : signed {
    EMPTY = -1,
    COLOR0 = GL_COLOR_ATTACHMENT0,
    COLOR1 = GL_COLOR_ATTACHMENT1,
    COLOR2 = GL_COLOR_ATTACHMENT2,
    COLOR3 = GL_COLOR_ATTACHMENT3,
    COLOR4 = GL_COLOR_ATTACHMENT4,
    COLOR5 = GL_COLOR_ATTACHMENT5,
    COLOR6 = GL_COLOR_ATTACHMENT6,
    COLOR7 = GL_COLOR_ATTACHMENT7,
    DEPTH = GL_DEPTH_ATTACHMENT,
    STENCIL = GL_STENCIL_ATTACHMENT,
    DEPTH_STENCIL = GL_DEPTH_STENCIL_ATTACHMENT
  };

  enum TEXTURE_TARGET : unsigned {
    TEXTURE2D = GL_TEXTURE_2D,
    CUBEMAP_POSITIVE_X = GL_TEXTURE_CUBE_MAP_POSITIVE_X,
    CUBEMAP_NEGATIVE_X = GL_TEXTURE_CUBE_MAP_NEGATIVE_X,
    CUBEMAP_POSITIVE_Y = GL_TEXTURE_CUBE_MAP_POSITIVE_Y,
    CUBEMAP_NEGATIVE_Y = GL_TEXTURE_CUBE_MAP_NEGATIVE_Y,
    CUBEMAP_POSITIVE_Z = GL_TEXTURE_CUBE_MAP_POSITIVE_Z,
    CUBEMAP_NEGATIVE_Z = GL_TEXTURE_CUBE_MAP_NEGATIVE_Z
  };

 protected:
  TEXTURE_TARGET target_texture_type{};
  std::unique_ptr<GLRenderBuffer> renderbuffer_object{};
  unsigned int framebuffer_id{};
  unsigned int texture_id{};
  unsigned int *pointer_on_default_fbo_id{};  // Needed for QT, since it's default framebuffer isn't 0. Bad ... but necessary for now.

 public:
  GLFrameBuffer();
  GLFrameBuffer(unsigned width,
                unsigned height,
                GLRenderBuffer::INTERNAL_FORMAT renderbuffer_internal_format = GLRenderBuffer::EMPTY,
                unsigned int *default_fbo_id_pointer = nullptr,
                TEXTURE_TARGET target_texture_type = TEXTURE2D);

  void initialize() override;
  ax_no_discard bool isReady() const override;
  void attachTexture2D(INTERNAL_FORMAT color_attachment, TEXTURE_TARGET target, unsigned int texture_id, unsigned int mipmap_level = 0);
  void bind() override;
  void unbind() override;
  void clean() override;
  void resize(unsigned int width, unsigned int height);

 private:
  void fill() override;
};

#endif