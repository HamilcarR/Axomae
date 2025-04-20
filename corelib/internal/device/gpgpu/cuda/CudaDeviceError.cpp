#include "../DeviceError.h"
#include <cuda_runtime_api.h>
#include <internal/common/exception/GenericException.h>
#include <internal/debug/Logger.h>

namespace exception {
  class WrongDeviceTypeException : public GenericException {
   public:
    explicit WrongDeviceTypeException(const std::string &err) : GenericException() { saveErrorString(err); }
  };
}  // namespace exception

DeviceError::DeviceError(cudaError_t error) : id(error) {}

bool DeviceError::isValid() const { return id == cudaSuccess; }

void gpgpu_err_log(cudaError_t code, const char *file, int line, bool abort) {
  if (code != cudaSuccess) {
    const char *err_str = cudaGetErrorString(code);
    std::string err = "CUDA GPU assert , error : " + std::to_string(code) + " " + std::string(err_str);
    LOGS(err + "\n File: " + file + "\n Line: " + std::to_string(line));
    if (abort)
      exit(code);
  }
}

void gpgpu_err_log(const DeviceError &err, const char *file, int line, bool abort) {
  cudaError_t code = err.getId();
  gpgpu_err_log(code, file, line, abort);
}

