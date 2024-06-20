#include "Camera.h"
#include "CameraFrameBuffer.h"
#include "NovaRenderer.h"
#include "RenderPipeline.h"
#include "Scene.h"
void NovaRenderer::setGammaValue(float value) {
  if (camera_framebuffer) {
    camera_framebuffer->setGamma(value);
  }
}

void NovaRenderer::setExposureValue(float value) {
  if (camera_framebuffer) {
    camera_framebuffer->setExposure(value);
  }
}

void NovaRenderer::setNoPostProcess() {
  if (camera_framebuffer) {
    camera_framebuffer->setPostProcessDefault();
  }
}

void NovaRenderer::setPostProcessEdge() {
  if (camera_framebuffer) {
    camera_framebuffer->setPostProcessEdge();
  }
}

void NovaRenderer::setPostProcessSharpen() {
  if (camera_framebuffer) {
    camera_framebuffer->setPostProcessSharpen();
  }
}

void NovaRenderer::setPostProcessBlurr() {
  if (camera_framebuffer) {
    camera_framebuffer->setPostProcessBlurr();
  }
}

void NovaRenderer::resetSceneCamera() {
  if (scene_camera) {
    scene_camera->reset();
  }
}

void NovaRenderer::setRasterizerFill() {
  if (scene) {
    scene->setPolygonFill();
  }
}

void NovaRenderer::setRasterizerPoint() {
  if (scene) {
    scene->setPolygonPoint();
  }
}

void NovaRenderer::setRasterizerWireframe() {
  if (scene) {
    scene->setPolygonWireframe();
  }
}

void NovaRenderer::displayBoundingBoxes(bool display) {
  if (scene) {
    scene->displayBoundingBoxes(display);
  }
}
void NovaRenderer::setViewerWidget(GLViewer *widget) {
  gl_widget = widget;
  if (render_pipeline)
    render_pipeline->setContextSwitcher(widget);
}
