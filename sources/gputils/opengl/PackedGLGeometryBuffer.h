#ifndef GLGEOMETRYBUFFER_H
#define GLGEOMETRYBUFFER_H
#include "../DeviceBufferInterface.h"
#include "Object3D.h"

#include "GLIndexBufferObject.h"
#include "GLVertexArrayObject.h"
#include "GLVertexBufferObject.h"

/**
 * Wrapper for opengl buffers functions , related to geometry and vertices attributes
 */

/**
 * @brief Wrapper class for Opengl vertices attributes buffers
 */
class PackedGLGeometryBuffer : public DeviceMutableBufferInterface {
 private:
  GLVertexArrayObject vao{};
  GLVertexBufferObject<float> vertex_buffer{};
  GLVertexBufferObject<float> normal_buffer{};
  GLVertexBufferObject<float> texture_buffer{};
  GLVertexBufferObject<float> color_buffer{};
  GLVertexBufferObject<float> tangent_buffer{};
  GLIndexBufferObject index_buffer{};
  const Object3D *geometry{};
  bool buffers_filled{false};

 public:
  PackedGLGeometryBuffer();
  explicit PackedGLGeometryBuffer(const Object3D *geometry);
  virtual void setGeometryPointer(const Object3D *geo) { geometry = geo; };
  /**
   * @brief Initialize glGenBuffers for all vertex attributes
   */
  void initialize() override;
  bool isReady() const override;
  void clean() override;
  void bind() override;
  void unbind() override;
  void fill() override;

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
