#ifndef RENDERQUAD_H
#define RENDERQUAD_H

#include "IFrameBuffer.h"

class RenderQuadFBO : public IFrameBuffer {
 public:
  RenderQuadFBO();
  RenderQuadFBO(TextureDatabase *texture_database, Dim2 *texture_size, unsigned int *default_fbo_id);
  virtual void renderToTexture(GLFrameBuffer::INTERNAL_FORMAT internal_format = GLFrameBuffer::COLOR0);
};

#endif