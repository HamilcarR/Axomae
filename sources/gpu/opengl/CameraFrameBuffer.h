#ifndef CAMERAFRAMEBUFFER_H
#define CAMERAFRAMEBUFFER_H

#include "IFrameBuffer.h"
#include "ResourceDatabaseManager.h"

/**
 * @file CameraFrameBuffer.h
 * This file implements the post processing framebuffer system
 */
class Mesh;
class Drawable;

/**
 * @class CameraFrameBuffer
 */
class CameraFrameBuffer : public IFrameBuffer {
 protected:
  std::unique_ptr<Drawable> drawable_screen_quad; /*<Drawable of the screen quad*/
  Mesh *mesh_screen_quad;                         /*<Pointer on the screen quad mesh*/
  ShaderDatabase *shader_database;                /*<Pointer on the shader database*/
  TextureDatabase *texture_database;
  INodeDatabase *node_database;
  ScreenFramebufferShader *shader_framebuffer; /*<Post processing shader*/
  float gamma;                                 /*<Gamma of the screen*/
  float exposure;                              /*<Exposure of the screen*/

 public:
  explicit CameraFrameBuffer(ResourceDatabaseManager &resource_database, Dim2 *screen_size_pointer, unsigned int *default_fbo_id);
  virtual void renderFrameBufferMesh();
  Drawable *getDrawable() { return drawable_screen_quad.get(); }
  /**
   * @brief Initializes the textures used by the framebuffer , the shader , and creates the quad mesh that the
   * framebuffer will draw on
   */
  virtual void initializeFrameBuffer() override;
  /**
   * @brief Send the uniforms used by the post processing effects , like gamma and exposure , and sets up the mesh used
   */
  virtual void startDraw();
  virtual void clean() override;
  virtual void updateFrameBufferShader();
  void setGamma(float value) { gamma = value; }
  void setExposure(float value) { exposure = value; }
  void setPostProcessDefault();
  void setPostProcessEdge();
  void setPostProcessSharpen();
  void setPostProcessBlurr();
};

#endif