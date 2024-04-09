
#include "FreePerspectiveCamera.h"

FreePerspectiveCamera::FreePerspectiveCamera() : Camera() { type = PERSPECTIVE; }

FreePerspectiveCamera::FreePerspectiveCamera(float deg, Dim2 *screen, float near, float far) : Camera(deg, near, far, screen) { type = PERSPECTIVE; }

void FreePerspectiveCamera::processEvent(const controller::event::Event *event) { EMPTY_FUNCBODY; }

void FreePerspectiveCamera::zoomIn() {}

void FreePerspectiveCamera::zoomOut() {}

const glm::mat4 &FreePerspectiveCamera::getSceneTranslationMatrix() const { return glm::mat4(1.f); }

const glm::mat4 &FreePerspectiveCamera::getSceneRotationMatrix() const { return glm::mat4(1.f); }

void FreePerspectiveCamera::focus(const glm::vec3 &position) { EMPTY_FUNCBODY; }