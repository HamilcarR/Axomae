#include "../includes/FrameBufferInterface.h"

int FrameBufferInterface::setUpEmptyTexture(unsigned width,
                                            unsigned height,
                                            bool persistence,
                                            Texture::FORMAT internal_format,
                                            Texture::FORMAT data_format,
                                            Texture::FORMAT data_type,
                                            Texture::TYPE type,
                                            unsigned int mipmaps) {
  TextureData temp_empty_data_texture;
  temp_empty_data_texture.width = width;
  temp_empty_data_texture.height = height;
  temp_empty_data_texture.data = nullptr;
  temp_empty_data_texture.f_data = nullptr;
  temp_empty_data_texture.internal_format = internal_format;
  temp_empty_data_texture.data_format = data_format;
  temp_empty_data_texture.data_type = data_type;
  temp_empty_data_texture.mipmaps = mipmaps;
  return texture_database->addTexture(&temp_empty_data_texture, type, persistence);
}

FrameBufferInterface::FrameBufferInterface() {
  gl_framebuffer_object = nullptr;
  texture_dim = nullptr;
  texture_database = nullptr;
  default_framebuffer_pointer = nullptr;
}

FrameBufferInterface::FrameBufferInterface(TextureDatabase *_texture_database,
                                           ScreenSize *_texture_dim,
                                           unsigned int *default_fbo)
    : FrameBufferInterface() {
  texture_dim = _texture_dim;
  texture_database = _texture_database;
  default_framebuffer_pointer = default_fbo;
  assert(texture_dim != nullptr);
}

FrameBufferInterface::~FrameBufferInterface() {
  if (gl_framebuffer_object != nullptr)
    delete gl_framebuffer_object;
}

void FrameBufferInterface::resize() {
  if (texture_dim && gl_framebuffer_object) {
    for (auto A : fbo_attachment_texture_collection) {
      A.second->setNewSize(texture_dim->width, texture_dim->height);
      gl_framebuffer_object->resize(texture_dim->width, texture_dim->height);
    }
  }
}

void FrameBufferInterface::bindFrameBuffer() {
  if (gl_framebuffer_object)
    gl_framebuffer_object->bind();
}

void FrameBufferInterface::unbindFrameBuffer() {
  if (gl_framebuffer_object)
    gl_framebuffer_object->unbind();
}

void FrameBufferInterface::clean() {
  if (gl_framebuffer_object)
    gl_framebuffer_object->clean();
}

void FrameBufferInterface::initializeFrameBufferTexture(GLFrameBuffer::INTERNAL_FORMAT color_attachment,
                                                        bool persistence,
                                                        Texture::FORMAT internal_format,
                                                        Texture::FORMAT data_format,
                                                        Texture::FORMAT data_type,
                                                        unsigned width,
                                                        unsigned height,
                                                        Texture::TYPE render_type,
                                                        unsigned int mipmaps) {

  unsigned int texture_id = setUpEmptyTexture(
      width, height, persistence, internal_format, data_format, data_type, render_type, mipmaps);
  fbo_attachment_texture_collection[color_attachment] = texture_database->get(texture_id);
}

void FrameBufferInterface::initializeFrameBuffer() {
  gl_framebuffer_object = new GLFrameBuffer(
      texture_dim->width, texture_dim->height, GLRenderBuffer::DEPTH24_STENCIL8, default_framebuffer_pointer);
  gl_framebuffer_object->initializeBuffers();
}