#ifndef DRAWABLE_H
#define DRAWABLE_H

#include "Camera.h"
#include "GLGeometryBuffer.h"
#include "LightingDatabase.h"
#include "Mesh.h"
#include "TextureGroup.h"

/**
 * @file Drawable.h
 * Implements a wrapper containing a mesh structure , a reference to a camera , and opengl buffers
 */

/***
 * @brief OpenGL structures relative to drawing one mesh
 * Manages API calls
 */
class Drawable {
 protected:
  Mesh *mesh_object;           /**<Pointer to the mesh */
  Camera *camera_pointer;      /**<Pointer to the camera*/
  GLGeometryBuffer gl_buffers; /**<OpenGL buffers*/

 public:
  Drawable();
  /**
   * @brief Construct a new Drawable object from a Mesh.
   * !NOTE : The mesh needs to be fully constructed before linking it to a Drawable . Shaders in particular should be
   * set .
   */
  explicit Drawable(Mesh *mesh);
  bool initialize();
  void startDraw();
  void clean();
  void bind();
  void unbind();
  bool ready();
  void setSceneCameraPointer(Camera *camera);
  Mesh *getMeshPointer() { return mesh_object; }
  [[nodiscard]] Shader *getMeshShaderPointer() const;
  [[nodiscard]] GLMaterial *getMaterialPointer() const;
};

#endif
