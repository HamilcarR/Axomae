#ifndef GLVERTEXARRAYOBJECT_H
#define GLVERTEXARRAYOBJECT_H
#include "../DeviceBufferInterface.h"
#include "project_macros.h"
#include <GL/glew.h>

class GLVertexArrayObject : public DeviceBaseBufferInterface {
 private:
  GLuint id{0};

 public:
  CLASS_OCM(GLVertexArrayObject)
  void initialize() override;
  [[nodiscard]] bool isReady() const override;
  void bind() override;
  void unbind() override;
  void clean() override;
};

#endif  // GLVERTEXARRAYOBJECT_H
