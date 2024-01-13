#ifndef RENDERCUBEMAP_H
#define RENDERCUBEMAP_H

#include "FrameBufferInterface.h"
#include "Shader.h"

/**
 * @brief A Framebuffer that renders to a cubemap
 *
 */
class RenderCubeMap : public FrameBufferInterface {
 public:
  RenderCubeMap();
  RenderCubeMap(TextureDatabase *texture_database, Dim2 *texture_size, unsigned int *default_fbo_pointer_id);
  virtual void renderToTexture(unsigned face = 0, GLFrameBuffer::INTERNAL_FORMAT color_attachment = GLFrameBuffer::COLOR0, unsigned mipmap_level = 0);

 protected:
};

#endif