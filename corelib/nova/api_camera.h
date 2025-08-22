#ifndef API_CAMERA_H
#define API_CAMERA_H
#include "api_common.h"

namespace nova {
  class Camera {
   public:
    virtual ~Camera() = default;
    virtual ERROR_STATE setUpVector(const float up[3]) = 0;
    virtual ERROR_STATE setMatrices(const float projection_matrix[16], const float view_matrix[16]) = 0;
    virtual ERROR_STATE setPosition(const float pos[3]) = 0;
    virtual ERROR_STATE setDirection(const float dir[3]) = 0;
    virtual ERROR_STATE setClipPlaneFar(float far_plane) = 0;
    virtual ERROR_STATE setClipPlaneNear(float near_plane) = 0;
    virtual ERROR_STATE setResolutionWidth(unsigned width) = 0;
    virtual ERROR_STATE setResolutionHeight(unsigned height) = 0;
    virtual ERROR_STATE setFov(float fov) = 0;

    virtual ERROR_STATE getUpVector(float up[3]) const = 0;
    virtual ERROR_STATE getProjectionMatrix(float proj[16]) const = 0;
    virtual ERROR_STATE getViewMatrix(float view[16]) const = 0;
    virtual ERROR_STATE getInverseProjectionMatrix(float proj[16]) const = 0;
    virtual ERROR_STATE getInverseViewMatrix(float proj[16]) const = 0;
    virtual ERROR_STATE getProjectionViewMatrix(float proj[16]) const = 0;
    virtual ERROR_STATE getInverseProjectionViewMatrix(float proj[16]) const = 0;
    virtual ERROR_STATE getPosition(float pos[3]) const = 0;
    virtual ERROR_STATE getDirection(float dir[3]) const = 0;
    virtual float getClipPlaneFar() const = 0;
    virtual float getClipPlaneNear() const = 0;
    virtual unsigned getResolutionWidth() const = 0;
    virtual unsigned getResolutionHeight() const = 0;
    virtual float getFov() const = 0;
  };

}  // namespace nova

#endif
