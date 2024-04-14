#ifndef GLGEOMETRYBUFFER_H
#define GLGEOMETRYBUFFER_H
#include "GLBufferInterface.h"
#include "Object3D.h"
#include "init_3D.h"
/**
 * @file GLGeometryBuffer.h
 * Wrapper for opengl buffers functions , related to geometry and vertices attributes
 */

/**
 * @brief Wrapper class for Opengl vertices attributes buffers
 */
class GLGeometryBuffer : public GLBufferInterface {
 private:
  GLuint vao;               /**<VAO ID*/
  GLuint vertex_buffer;     /**<Vertex buffer ID*/
  GLuint normal_buffer;     /**<Normal buffer ID*/
  GLuint index_buffer;      /**<Index buffer ID*/
  GLuint texture_buffer;    /**<Texture buffer ID*/
  GLuint color_buffer;      /**<Color buffer ID*/
  GLuint tangent_buffer;    /**<Tangent buffer ID*/
  const Object3D *geometry; /**<Pointer to the meshe's geometry*/
  bool buffers_filled;

 public:
  GLGeometryBuffer();
  explicit GLGeometryBuffer(const Object3D *geometry);
  virtual void setGeometryPointer(const Object3D *geo) { geometry = geo; };
  /**
   * @brief Initialize glGenBuffers for all vertex attributes
   */
  void initializeBuffers() override;
  bool isReady() const override;
  void clean() override;
  void bind() override;
  void unbind() override;
  void fillBuffers() override;

  void bindVao();
  void unbindVao();
  void bindVertexBuffer();
  void bindNormalBuffer();
  void bindTextureBuffer();
  void bindColorBuffer();
  void bindIndexBuffer();
  void bindTangentBuffer();
};

#endif
