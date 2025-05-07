#ifndef NOVA_GPU_H
#define NOVA_GPU_H
#include "engine/datastructures.h"

namespace nova {

  struct device_random_generators_s {
    math::random::GPUQuasiRandomGenerator sobol;
  };

  /* This structure is used as pipeline parameter for optix and other APIs.*/
  struct device_traversal_param_s {
    HdrBufferStruct render_buffers;
    shape::MeshBundleViews mesh_bundle_views;
    texturing::TextureBundleViews texture_bundle_views;
    primitive::CstPrimitivesView primitives_view;
    material::CstNovaMatIntfView material_view;
    texturing::EnvmapTexture environment_map;
    camera::CameraResourcesHolder camera;
    device_random_generators_s device_random_generators;
    unsigned width, height, depth;
  };

}  // namespace nova
#endif  // NOVA_GPU_H
