#ifndef DEVICESHADERINTERFACE_H
#define DEVICESHADERINTERFACE_H
#include <string>

class DeviceShaderInterface {
 public:
  virtual ~DeviceShaderInterface() = default;
  virtual void initializeShader() = 0;
  virtual void recompile() = 0;
  virtual void bind() = 0;
  virtual void release() = 0;
  virtual void clean() = 0;
  virtual void setShadersRawText(const std::string &vs, const std::string &fs) = 0;
  [[nodiscard]] virtual bool isInitialized() const = 0;
};

#endif
