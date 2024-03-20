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
 *
 */

/***
 * @brief OpenGL structures relative to drawing one mesh
 * Manages API calls
 */
class Drawable {
 public:
  /**
   * @brief Construct a new Drawable object
   *
   */
  Drawable();

  /**
   * @brief Construct a new Drawable object from a Mesh.
   * !NOTE : The mesh needs to be fully constructed before linking it to a Drawable . Shaders in particular should be
   * set .
   * @param mesh Pointer to a mesh
   */
  Drawable(Mesh *mesh);

  /**
   * @brief Initialize gl buffers and Mesh data like materials and shaders
   *
   * @return true If operation succeeds
   */
  bool initialize();

  /**
   * @brief Binds shaders ,and buffers
   *
   */
  void startDraw();

  /**
   * @brief Cleans buffers , and mesh data
   *
   */
  void clean();

  /**
   * @brief Binding before draw
   *
   */
  void bind();

  /**
   * @brief Unbind after draw
   *
   */
  void unbind();

  /**
   * @brief Checks if ready to draw
   *
   * @return true if is ready to draw
   */
  bool ready();

  /**
   * @brief Set the Scene Camera Pointer
   *
   * @param camera Camera pointer
   */
  void setSceneCameraPointer(Camera *camera);

  /**
   * @brief Returns the mesh
   *
   */
  Mesh *getMeshPointer() { return mesh_object; }

  /**
   * @brief Get the Mesh's Shader pointer
   *
   * @return Shader* Shader pointer , or nullptr if mesh empty
   */
  Shader *getMeshShaderPointer() const;

  /**
   * @brief Get the mesh material pointer
   *
   * @return Material*
   */
  GLMaterial *getMaterialPointer() const;

 protected:
  Mesh *mesh_object;           /**<Pointer to the mesh */
  Camera *camera_pointer;      /**<Pointer to the camera*/
  GLGeometryBuffer gl_buffers; /**<OpenGL buffers*/
};

#endif
