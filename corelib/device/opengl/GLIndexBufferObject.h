
#ifndef GLINDEXBUFFEROBJECT_H
#define GLINDEXBUFFEROBJECT_H
#include "../DeviceBufferInterface.h"
#include "init_3D.h"
#include "project_macros.h"
class GLIndexBufferObject : public DeviceBaseBufferInterface {
 public:
  enum DRAW_MODE : int { STATIC = GL_STATIC_DRAW, DYNAMIC = GL_DYNAMIC_DRAW, STREAM = GL_STREAM_DRAW };

 private:
  GLuint id;

 public:
  CLASS_OCM(GLIndexBufferObject)

  void initialize() override;
  [[nodiscard]] bool isReady() const override;
  void bind() override;
  void unbind() override;
  void clean() override;
  void fill(const unsigned *buffer, size_t number_elements, DRAW_MODE mode);
};

#endif  // GLINDEXBUFFEROBJECT_H
