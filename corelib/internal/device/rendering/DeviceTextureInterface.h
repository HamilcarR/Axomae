#ifndef DEVICETEXTUREINTERFACE_H
#define DEVICETEXTUREINTERFACE_H
#include "internal/macro/project_macros.h"
class DeviceTextureInterface {
 public:
  virtual ~DeviceTextureInterface() = default;
  ax_no_discard virtual bool isInitialized() const = 0;
  virtual void clean() = 0;
};

#endif
