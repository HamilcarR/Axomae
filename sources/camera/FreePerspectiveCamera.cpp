
#include "FreePerspectiveCamera.h"

FreePerspectiveCamera::FreePerspectiveCamera() : Camera() { type = PERSPECTIVE; }

FreePerspectiveCamera::FreePerspectiveCamera(float deg, Dim2 *screen, float near, float far) : Camera(deg, near, far, screen) { type = PERSPECTIVE; }

void FreePerspectiveCamera::processEvent(const controller::event::Event *event) { EMPTY_FUNCBODY; }

void FreePerspectiveCamera::zoomIn() {}

void FreePerspectiveCamera::zoomOut() {}

const glm::mat4 &FreePerspectiveCamera::getSceneTranslationMatrix() const { AX_UNREACHABLE; }

const glm::mat4 &FreePerspectiveCamera::getSceneRotationMatrix() const { AX_UNREACHABLE; }

void FreePerspectiveCamera::focus(const glm::vec3 &position) { EMPTY_FUNCBODY; }