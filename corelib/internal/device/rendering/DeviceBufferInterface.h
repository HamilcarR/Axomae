#ifndef DEVICEBUFFERINTERFACE_H
#define DEVICEBUFFERINTERFACE_H
#include <internal/macro/project_macros.h>

/**
 * Interface modeling a device's allocated memory.
 */

class DeviceBaseBufferInterface {
 public:
  virtual ~DeviceBaseBufferInterface() = default;
  virtual void initialize() = 0;
  ax_no_discard virtual bool isReady() const = 0;
  virtual void bind() = 0;
  virtual void unbind() = 0;
  virtual void clean() = 0;
  // TODO : add - virtual uint32_t getID() const = 0;
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