#include "DeviceError.h"

#include <GenericException.h>

namespace exception {
  class WrongDeviceTypeException : public GenericException {
   public:
    explicit WrongDeviceTypeException(const std::string &err) : GenericException() { saveErrorString(err); }
  };
}  // namespace exception

DeviceError::DeviceError(cudaError_t error) : err(error) {}

bool DeviceError::check(DEVICETYPE type) const {
  switch (type) {
    case CUDA:
      return err == 0;
    default:
      throw exception::WrongDeviceTypeException("Unknown Device error type used.");
  }
}
