#ifndef RENDERERENUMS_H
#define RENDERERENUMS_H

enum RENDERER_CALLBACK_ENUM : unsigned {
  SET_GAMMA,
  SET_EXPOSURE,
  SET_POSTPROCESS_NOPROCESS,
  SET_POSTPROCESS_SHARPEN,
  SET_POSTPROCESS_BLURR,
  SET_POSTPROCESS_EDGE,
  SET_RASTERIZER_POINT,
  SET_RASTERIZER_FILL,
  SET_RASTERIZER_WIREFRAME,
  SET_DISPLAY_BOUNDINGBOX,
  SET_DISPLAY_RESET_CAMERA,
  ADD_ELEMENT_POINTLIGHT  // TODO : create light builder
};

#endif