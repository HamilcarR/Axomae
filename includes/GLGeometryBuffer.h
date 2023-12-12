#ifndef GLGEOMETRYBUFFER_H
#define GLGEOMETRYBUFFER_H
#include "GLBufferInterface.h"

/**
 * @file GLGeometryBuffer.h
 * Wrapper for opengl buffers functions , related to geometry and vertices attributes
 *
 */

/**
 * @brief Wrapper class for Opengl vertices attributes buffers
 *
 */
class GLGeometryBuffer : public GLBufferInterface {
 public:
  /**
   * @brief Construct a new GLGeometryBuffer object
   *
   */
  GLGeometryBuffer();

  /**
   * @brief Construct a new GLGeometryBuffer object
   *
   * @param geometry Mesh geometry data pointer
   * @see Object3D
   */
  GLGeometryBuffer(const Object3D *geometry);

  /**
   * @brief Destroy the GLGeometryBuffer object
   *
   */
  virtual ~GLGeometryBuffer();

  /**
   * @brief Sets the current geometry pointer
   *
   * @param geo Pointer on the current mesh geometry
   */
  virtual void setGeometryPointer(const Object3D *geo) { geometry = geo; };

  /**
   * @brief Initialize glGenBuffers for all vertex attributes
   *
   */
  virtual void initializeBuffers();

  /**
   * @brief Checks if VAOs and VBOs are initialized
   *
   * @return true If the buffers are ready to be used
   */
  virtual bool isReady() const;

  /**
   * @brief Delete GL buffers and VAOs of the mesh
   *
   */
  virtual void clean();

  /**
   * @brief Binds every initialized buffer
   *
   */
  virtual void bind();

  /**
   * @brief Unbinds every initialized buffer
   *
   */
  virtual void unbind();

  /**
   * @brief Binds the vertex array object of the mesh
   *
   */
  void bindVao();

  /**
   * @brief Unbind the vertex array object
   *
   */
  void unbindVao();

  /**
   * @brief Binds the vertex buffer object
   *
   */
  void bindVertexBuffer();

  /**
   * @brief Binds the normal buffer object
   *
   */
  void bindNormalBuffer();

  /**
   * @brief Binds the texture buffer object
   *
   */
  void bindTextureBuffer();

  /**
   * @brief Binds the color buffer object
   *
   */
  void bindColorBuffer();

  /**
   * @brief Binds the index buffer object
   *
   */
  void bindIndexBuffer();

  /**
   * @brief Binds the tangent buffer object
   *
   */
  void bindTangentBuffer();

  /**
   * @brief Transfers data to GPU using glBufferData on all vertex buffers
   *
   */
  virtual void fillBuffers();

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
};

#endif
