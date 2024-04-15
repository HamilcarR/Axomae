#ifndef RENDERCUBEMAP_H
#define RENDERCUBEMAP_H

#include "IFrameBuffer.h"
#include "Shader.h"

/**
 * @brief A Framebuffer that renders to a cubemap
 *
 */
class RenderCubeMap : public IFrameBuffer {
 public:
  RenderCubeMap() = default;
  ~RenderCubeMap() override = default;
  RenderCubeMap(const RenderCubeMap &copy) = delete;
  RenderCubeMap(RenderCubeMap &&move) noexcept = default;
  RenderCubeMap(TextureDatabase *texture_database, Dim2 *texture_size, unsigned int *default_fbo_pointer_id);
  virtual void renderToTexture(unsigned face, GLFrameBuffer::INTERNAL_FORMAT color_attachment, unsigned mipmap_level);

 protected:
};

#endif