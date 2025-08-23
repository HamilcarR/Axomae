#ifndef API_COMMON_H
#define API_COMMON_H
#include "api_datastructures.h"
#include <internal/common/axstd/span.h>
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
    INVALID_ARGUMENT,
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

      /* Render passes. */
      COMBINED = 1 << 8,
      NORMAL = 1 << 9,
      DEPTH = 1 << 10,
      SPECULAR = 1 << 11,
      DIFFUSE = 1 << 12,
      EMISSIVE = 1 << 13,
    };

  }

  namespace texture {
    enum FORMAT {
      UINT8X4,
      FLOATX4,
    };

  }

  namespace mesh {
    // Only triangle is supported for now.
    enum TYPE { TRIANGLE, SPHERE, BOX, NURB };

  }  // namespace mesh

  class Camera;
  class Scene;
  class Transform;
  class Engine;
  class Trimesh;
  class Material;
  class Texture;
  class RenderBuffer;
  class RenderOptions;

  using RenderBufferPtr = std::unique_ptr<RenderBuffer>;
  using EnginePtr = std::unique_ptr<Engine>;
  using RenderOptionsPtr = std::unique_ptr<RenderOptions>;
  using TransformPtr = std::unique_ptr<Transform>;
  using TexturePtr = std::unique_ptr<Texture>;
  using ScenePtr = std::unique_ptr<Scene>;
  using TrimeshPtr = std::unique_ptr<Trimesh>;
  using CameraPtr = std::unique_ptr<Camera>;
  using MaterialPtr = std::unique_ptr<Material>;
  using CsteTriMeshCollection = axstd::span<const TrimeshPtr>;
  using CsteCameraCollection = axstd::span<const CameraPtr>;
  using CsteMaterialCollection = axstd::span<const MaterialPtr>;
  using CsteTextureCollection = axstd::span<const TexturePtr>;

}  // namespace nova
#endif
