#include "CudaParams.h"

#include <GenericException.h>

namespace exception {
  class InvalidCudaMemcpyKindException : public GenericException {
   public:
    explicit InvalidCudaMemcpyKindException(const std::string &err) : GenericException() { saveErrorString(err); }
  };
}  // namespace exception

namespace ax_cuda {

  void CudaParams::setMemcpyKind(unsigned copy_kind) {
    switch (copy_kind) {
      case cudaMemcpyHostToHost:
        memcpy_kind = cudaMemcpyHostToHost;
        break;
      case cudaMemcpyHostToDevice:
        memcpy_kind = cudaMemcpyHostToDevice;
        break;
      case cudaMemcpyKind::cudaMemcpyDeviceToDevice:
        memcpy_kind = cudaMemcpyDeviceToDevice;
        break;
      case cudaMemcpyKind::cudaMemcpyDeviceToHost:
        memcpy_kind = cudaMemcpyHostToHost;
        break;
      case cudaMemcpyKind::cudaMemcpyDefault:
        memcpy_kind = cudaMemcpyDefault;
        break;
      default:
        std::string err = "Invalid memcpy kind.";
        LOG(err, LogLevel::ERROR);
        throw exception::InvalidCudaMemcpyKindException(err);
    }
  }

}  // namespace ax_cuda