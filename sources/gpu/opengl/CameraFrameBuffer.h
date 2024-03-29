#ifndef CAMERAFRAMEBUFFER_H
#define CAMERAFRAMEBUFFER_H

#include "Drawable.h"
#include "FrameBufferInterface.h"
#include "ResourceDatabaseManager.h"

/**
 * @file CameraFrameBuffer.h
 * This file implements the post processing framebuffer system
 */

/**
 * @class CameraFrameBuffer
 * This class implements post processing effects
 */
class CameraFrameBuffer : public FrameBufferInterface {
 public:
  /**
   * @brief Construct a new Camera Frame Buffer object
   * @param resource_database
   * @param screen_size_pointer Pointer on a Dim2 structure , containing informations about the dimensions of the
   * render surface
   * @param default_fbo_id Pointer on the ID of the default framebuffer . In the case of QT , this framebuffer is the
   * one used for the GUI interface (so , not 0)
   */
  CameraFrameBuffer(ResourceDatabaseManager &resource_database, Dim2 *screen_size_pointer, unsigned int *default_fbo_id);

  /**
   * @brief Destroy the Camera Frame Buffer object
   *
   */
  virtual ~CameraFrameBuffer();

  /**
   * @brief Renders the quad mesh of the framebuffer
   *
   */
  virtual void renderFrameBufferMesh();

  /**
   * @brief Returns a pointer on the drawable of the quad mesh
   *
   * @return Drawable*
   */
  Drawable *getDrawable() { return drawable_screen_quad.get(); }

  /**
   * @brief Initializes the textures used by the framebuffer , the shader , and creates the quad mesh that the
   * framebuffer will draw on
   *
   */
  virtual void initializeFrameBuffer() override;

  /**
   * @brief Send the uniforms used by the post processing effects , like gamma and exposure , and sets up the mesh used
   *
   */
  virtual void startDraw();

  /**
   * @brief Clean the post processing structure entirely , freeing the ressources
   *
   */
  virtual void clean() override;

  /**
   * @brief Update the shader used for post processing
   *
   */
  virtual void updateFrameBufferShader();

  /**
   * @brief Set the Gamma value
   *
   * @param value
   */
  void setGamma(float value) { gamma = value; }

  /**
   * @brief Set the Exposure value
   *
   * @param value
   */
  void setExposure(float value) { exposure = value; }

  /**
   * @brief Set the Post Process Default object
   *
   */
  void setPostProcessDefault();

  /**
   * @brief Set the Post Process Edge object
   *
   */
  void setPostProcessEdge();

  /**
   * @brief Set the Post Process Sharpen object
   *
   */
  void setPostProcessSharpen();

  /**
   * @brief Set the Post Process Blurr object
   *
   */
  void setPostProcessBlurr();

 private:
  /**
   * @brief Construct a new Camera Frame Buffer object
   *
   */
  CameraFrameBuffer();

 protected:
  std::unique_ptr<Drawable> drawable_screen_quad; /*<Drawable of the screen quad*/
  Mesh *mesh_screen_quad;                         /*<Pointer on the screen quad mesh*/
  ShaderDatabase *shader_database;                /*<Pointer on the shader database*/
  TextureDatabase *texture_database;
  INodeDatabase *node_database;
  ScreenFramebufferShader *shader_framebuffer; /*<Post processing shader*/
  float gamma;                                 /*<Gamma of the screen*/
  float exposure;                              /*<Exposure of the screen*/
};

#endif