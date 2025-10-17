#ifndef NOVA_GPU_H
#define NOVA_GPU_H
#include "camera/nova_camera.h"
#include "engine/datastructures.h"
#include "material/NovaMaterials.h"
#include "primitive/PrimitiveInterface.h"
#include "shape/MeshContext.h"
#include "texturing/TextureContext.h"
#include <internal/common/math/gpu/math_random_gpu.h>

namespace nova {

  struct device_random_generators_s {
    math::random::SobolGenerator rqmc_generator;
  };

  /* This structure is used as pipeline parameter for optix and other APIs.
   * Alignment is fixed to 8 bytes to  avoid memory corruption on device side.
   */
  struct device_traversal_param_s {
    HdrBufferStruct render_buffers;
    device_random_generators_s device_random_generators;
    shape::MeshBundleViews mesh_bundle_views;
    texturing::TextureBundleViews texture_bundle_views;
    primitive::CstPrimitivesView primitives_view;
    material::CstNovaMatIntfView material_view;
    camera::CameraResourcesHolder camera;
    uint32_t width, height, depth, current_envmap_index, sample_index, sample_max;
  };

}  // namespace nova

#endif  // NOVA_GPU_H
