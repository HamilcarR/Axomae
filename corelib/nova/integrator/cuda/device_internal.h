#ifndef DEVICE_INTERNAL_H
#define DEVICE_INTERNAL_H
#include "engine/datastructures.h"
#include "primitive/PrimitiveInterface.h"
#include "shape/ShapeInterface.h"
#include "texturing/TextureContext.h"

/* Structures living in gpu memory. */

struct render_buffer_s {
  float *render_target;
  unsigned width;
  unsigned height;
};

#endif
