#include "RenderCubeMap.h"

RenderCubeMap::RenderCubeMap(TextureDatabase *database, Dim2 *screen, unsigned int *default_fbo) : FramebufferHelper(database, screen, default_fbo) {}

void RenderCubeMap::renderToTexture(unsigned i, GLFrameBuffer::INTERNAL_FORMAT color_attachment, unsigned mip_level) {
  GenericTexture *tex = fbo_attachment_texture_collection[color_attachment];
  if (tex && tex->isInitialized())
    gl_framebuffer_object->attachTexture2D(
        color_attachment, static_cast<GLFrameBuffer::TEXTURE_TARGET>(GLFrameBuffer::CUBEMAP_POSITIVE_X + i), tex->getSamplerID(), mip_level);
}
