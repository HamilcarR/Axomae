#ifndef DEVICE_INTERNAL_H
#define DEVICE_INTERNAL_H
#include "primitive/PrimitiveInterface.h"
#include "shape/ShapeInterface.h"
#include "texturing/TextureContext.h"

#include <nova_gpu_utils.h>
#include <optix_types.h>

/* Structures living in gpu memory. */

struct render_buffer_t {
  float *render_target;
  unsigned width;
  unsigned height;
};

struct integrator_args_s {
  nova::gputils::gpu_random_generator_t generator;
  nova::shape::MeshCtx geometry_context;
  nova::texturing::TextureCtx texture_context;
  axstd::span<const nova::material::NovaMaterialInterface> materials;
  axstd::span<const nova::primitive::NovaPrimitiveInterface> primitives;
  OptixTraversableHandle optix_traversable_handle;
};

struct _device_params_s_ {};

#endif
