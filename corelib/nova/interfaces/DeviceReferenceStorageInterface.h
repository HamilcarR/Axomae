#ifndef DEVICEREFERENCESTORAGEINTERFACE_H
#define DEVICEREFERENCESTORAGEINTERFACE_H
#include <cstdlib>

class DeviceReferenceStorageInterface {
  public:
   virtual ~DeviceReferenceStorageInterface() = default;
   virtual std::size_t size() const = 0;
  virtual void clear() = 0;
  virtual void allocate(std::size_t size) = 0;
  virtual void mapBuffers() = 0 ;
  virtual void mapResources() = 0 ;
  virtual void release() = 0 ;
  };




#endif //DEVICEREFERENCESTORAGEINTERFACE_H
