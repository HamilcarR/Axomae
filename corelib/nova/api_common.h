#ifndef API_COMMON_H
#define API_COMMON_H
#include <memory>
namespace nova {
  enum ERROR_STATE {
    SUCCESS,

    INVALID_ENGINE_STATE,
    INVALID_BUFFER_STATE,
    INVALID_CHANNEL_DESCRIPTOR,
    INVALID_SCENE_TYPE,
    INVALID_TRANSFORM_TYPE,
    SCENE_NOT_PROCESSED,
    NOT_GPU_BUILD,
    OUT_OF_MEMORY,
    MULTIPLE_INTEGRATORS_NOT_SUPPORTED,
    THREADPOOL_CREATION_ERROR,
    THREADPOOL_NOT_INITIALIZED,

  };

  namespace integrator {
    enum TYPE : int {
      PATH = 1 << 0,
      BIPATH = 1 << 1,
      SPECTRAL = 1 << 2,
      METROPOLIS = 1 << 3,
      PHOTON = 1 << 4,
      MARCHING = 1 << 5,
      HYBRID = 1 << 6,
      VOXEL = 1 << 7,

      /* utility render */
      COMBINED = 1 << 8,
      NORMAL = 1 << 9,
      DEPTH = 1 << 10,
      SPECULAR = 1 << 11,
      DIFFUSE = 1 << 12,
      EMISSIVE = 1 << 13,
    };

  }

}  // namespace nova
#endif
