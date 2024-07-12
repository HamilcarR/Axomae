#ifndef RENDERQUAD_H
#define RENDERQUAD_H

#include "FramebufferHelper.h"

class RenderQuadFBO : public FramebufferHelper {
 public:
  RenderQuadFBO() = default;
  ~RenderQuadFBO() override = default;
  RenderQuadFBO(const RenderQuadFBO &copy) = delete;
  RenderQuadFBO(RenderQuadFBO &&move) noexcept = default;
  RenderQuadFBO(TextureDatabase *texture_database, Dim2 *texture_size, unsigned int *default_fbo_id);
  virtual void renderToTexture(GLFrameBuffer::INTERNAL_FORMAT internal_format);
};

#endif