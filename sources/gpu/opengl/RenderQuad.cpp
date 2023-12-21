#include "RenderQuad.h"

RenderQuadFBO::RenderQuadFBO() {}

RenderQuadFBO::RenderQuadFBO(TextureDatabase *database, ScreenSize *screen, unsigned int *default_fbo)
    : FrameBufferInterface(database, screen, default_fbo) {
  LOG("initialized cubemap", LogLevel::INFO);
}

void RenderQuadFBO::renderToTexture(GLFrameBuffer::INTERNAL_FORMAT color_attachment) {
  Texture *tex = fbo_attachment_texture_collection[color_attachment];
  if (tex && tex->isInitialized())
    gl_framebuffer_object->attachTexture2D(
        color_attachment, static_cast<GLFrameBuffer::TEXTURE_TARGET>(GLFrameBuffer::TEXTURE2D), tex->getSamplerID());
}
