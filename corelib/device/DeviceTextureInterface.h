#ifndef DEVICETEXTUREINTERFACE_H
#define DEVICETEXTUREINTERFACE_H

class DeviceTextureInterface {
 public:
  virtual ~DeviceTextureInterface() = default;
  [[nodiscard]] virtual bool isInitialized() const = 0;
  virtual void clean() = 0;
};

#endif
