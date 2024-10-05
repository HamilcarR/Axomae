#ifndef DEVICEERROR_H
#define DEVICEERROR_H
#include "internal/macro/project_macros.h"
#include <cstdint>
#include <driver_types.h>

#if defined(AXOMAE_USE_CUDA)

#  define DEVICE_ERROR_CHECK(ans) \
    { \
      cuAssert((ans), __FILE__, __LINE__); \
    }
void cuAssert(cudaError_t err, const char *file, int line, bool abort = false);

#else
#  define DEVICE_ERROR_CHECK(ans)
#endif
#endif  // DEVICEERROR_H
