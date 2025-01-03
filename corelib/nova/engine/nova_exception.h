#ifndef EXCEPTION_H
#define EXCEPTION_H
#include "internal/device/gpgpu/device_utils.h"
#include "internal/macro/class_macros.h"
#include <atomic>
#include <cstdint>

#include <internal/macro/project_macros.h>
#include <vector>

namespace nova::exception {

  enum ERROR : uint64_t {
    NOERR = 0,
    INVALID_INTEGRATOR = 1 << 0,
    INVALID_RENDER_MODE = 1 << 1,

    INVALID_SAMPLER_DIM = 1 << 2,
    SAMPLER_INIT_ERROR = 1 << 3,
    SAMPLER_DOMAIN_EXHAUSTED = 1 << 4,
    SAMPLER_INVALID_ALLOC = 1 << 5,
    SAMPLER_INVALID_ARG = 1 << 6,

    INVALID_RENDERBUFFER_STATE = 1 << 7,
    INVALID_ENGINE_INSTANCE = 1 << 8,
    INVALID_RENDERBUFFER_DIM = 1 << 9,

    GENERAL_GPU_ERROR = 1ULL << 31,
    GENERAL_ERROR = 1ULL << 32,

  };

  class NovaException {
   private:
    uint64_t synchronized_err_flag{NOERR};  // Synchronized on device , to be read by host. Otherwise we can't read err_flag if cuda is used, since it
                                            // depends on architecture context.

#if defined(__CUDA_ARCH__) && defined(AXOMAE_USE_CUDA)
    cuda::atomic<uint64_t> err_flag{NOERR};
#else
    std::atomic<uint64_t> err_flag{NOERR};
#endif

   private:
    ax_device_callable void synchronizeErrFlag();

   public:
    ax_device_callable NovaException() = default;
    ax_device_callable NovaException(NovaException &&move) noexcept;
    ax_device_callable NovaException &operator=(NovaException &&move) noexcept;
    ax_device_callable NovaException(const NovaException &copy) noexcept;
    ax_device_callable NovaException &operator=(const NovaException &copy) noexcept;
    ax_device_callable ~NovaException() = default;

    ax_device_callable ax_no_discard bool errorCheck() const { return synchronized_err_flag != NOERR; }
    ax_device_callable ax_no_discard uint64_t getErrorFlag() const { return synchronized_err_flag; }
    ax_device_callable void addErrorType(uint64_t to_add);
    /* merges err_flag and other_error_flag , err_flag will now store it's previous errors + other_error_flag */
    ax_device_callable void merge(uint64_t other_error_flag);

    ax_host_only ax_no_discard std::vector<ERROR> getErrorList() const;
  };

}  // namespace nova::exception

#endif  // EXCEPTION_H
