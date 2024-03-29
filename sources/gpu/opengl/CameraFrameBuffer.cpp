#include "CameraFrameBuffer.h"
#include "INodeFactory.h"
#include "UniformNames.h"
using namespace axomae;

CameraFrameBuffer::CameraFrameBuffer(ResourceDatabaseManager &resource_database, Dim2 *screen_size_pointer, unsigned int *default_fbo_pointer)
    : FrameBufferInterface(resource_database.getTextureDatabase(), screen_size_pointer, default_fbo_pointer),
      shader_database(resource_database.getShaderDatabase()),
      texture_database(resource_database.getTextureDatabase()),
      node_database(resource_database.getNodeDatabase()) {
  gamma = 1.2f;
  exposure = 0.3f;
  gl_framebuffer_object = nullptr;
  drawable_screen_quad = nullptr;
  mesh_screen_quad = nullptr;
  shader_framebuffer = nullptr;
}

CameraFrameBuffer::~CameraFrameBuffer() {}

void CameraFrameBuffer::updateFrameBufferShader() {
  shader_framebuffer = static_cast<ScreenFramebufferShader *>(shader_database->get(Shader::SCREEN_FRAMEBUFFER));
  assert(mesh_screen_quad);
  mesh_screen_quad->setShader(shader_framebuffer);
}

void CameraFrameBuffer::initializeFrameBuffer() {
  initializeFrameBufferTexture<FrameBufferTexture>(
      GLFrameBuffer::COLOR0, true, Texture::RGBA16F, Texture::BGRA, Texture::UBYTE, texture_dim->width, texture_dim->height);
  shader_framebuffer = static_cast<ScreenFramebufferShader *>(shader_database->get(Shader::SCREEN_FRAMEBUFFER));
  Texture *fbo_texture = fbo_attachment_texture_collection[GLFrameBuffer::COLOR0];
  assert(texture_database->contains(fbo_texture).object);
  int database_texture_id = texture_database->contains(fbo_texture).id;
  auto result = database::node::store<FrameBufferMesh>(*node_database, true, database_texture_id, shader_framebuffer);
  mesh_screen_quad = result.object;
  drawable_screen_quad = std::make_unique<Drawable>(mesh_screen_quad);
  FrameBufferInterface::initializeFrameBuffer();
  bindFrameBuffer();
  gl_framebuffer_object->attachTexture2D(GLFrameBuffer::COLOR0, GLFrameBuffer::TEXTURE2D, fbo_texture->getSamplerID());
  unbindFrameBuffer();
}

void CameraFrameBuffer::clean() {
  if (drawable_screen_quad)
    drawable_screen_quad->clean();
  FrameBufferInterface::clean();
}

void CameraFrameBuffer::startDraw() {
  if (shader_framebuffer) {
    shader_framebuffer->bind();
    shader_framebuffer->setUniform(uniform_name_float_gamma_name, gamma);
    shader_framebuffer->setUniform(uniform_name_float_exposure_name, exposure);
    shader_framebuffer->setPostProcessUniforms();
    shader_framebuffer->release();
  }
  if (drawable_screen_quad)
    drawable_screen_quad->startDraw();
}

void CameraFrameBuffer::renderFrameBufferMesh() {
  drawable_screen_quad->bind();
  glDrawElements(GL_TRIANGLES, drawable_screen_quad->getMeshPointer()->geometry.indices.size(), GL_UNSIGNED_INT, 0);
  drawable_screen_quad->unbind();
}

void CameraFrameBuffer::setPostProcessEdge() {
  if (shader_framebuffer)
    shader_framebuffer->setPostProcess(ScreenFramebufferShader::EDGE);
}

void CameraFrameBuffer::setPostProcessDefault() {
  if (shader_framebuffer)
    shader_framebuffer->setPostProcess(ScreenFramebufferShader::DEFAULT);
}

void CameraFrameBuffer::setPostProcessBlurr() {
  if (shader_framebuffer)
    shader_framebuffer->setPostProcess(ScreenFramebufferShader::BLURR);
}

void CameraFrameBuffer::setPostProcessSharpen() {
  if (shader_framebuffer)
    shader_framebuffer->setPostProcess(ScreenFramebufferShader::SHARPEN);
}