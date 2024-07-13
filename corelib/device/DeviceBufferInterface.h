#ifndef DEVICEBUFFERINTERFACE_H
#define DEVICEBUFFERINTERFACE_H

/**
 * Interface modeling a device's allocated memory.
 */

class DeviceBaseBufferInterface {
 public:
  virtual ~DeviceBaseBufferInterface() = default;
  virtual void initialize() = 0;
  [[nodiscard]] virtual bool isReady() const = 0;
  virtual void bind() = 0;
  virtual void unbind() = 0;
  virtual void clean() = 0;
};

class DeviceMutableBufferInterface : public DeviceBaseBufferInterface {
 public:
  virtual void fill() = 0;
};

class DeviceImmutableBufferInterface : public DeviceBaseBufferInterface {
 public:
  virtual void fillStorage(const void *data) = 0;
};
#endif