#ifndef DRAWABLE_H
#define DRAWABLE_H
#include "PackedGLGeometryBuffer.h"
#include <internal/macro/project_macros.h>

class Camera;
class Mesh;
class GLMaterial;
class Shader;
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
  Mesh *mesh_object;                 /**<Pointer to the mesh */
  Camera *camera_pointer;            /**<Pointer to the camera*/
  PackedGLGeometryBuffer gl_buffers; /**<OpenGL buffers*/

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
  Mesh *getMeshPointer() const { return mesh_object; }
  ax_no_discard Shader *getMeshShaderPointer() const;
  ax_no_discard GLMaterial *getMaterialPointer() const;
};

#endif
