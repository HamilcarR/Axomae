#ifndef MESH_H
#define MESH_H

#include "Camera.h"
#include "Material.h"
#include "MeshInterface.h"
#include "internal/geometry/BoundingBox.h"
#include "internal/geometry/Object3D.h"

/**
 * @brief Mesh.h
 * Mesh class implementation
 */

/**
 * @brief Mesh class
 */
class Mesh : public SceneTreeNode, public MeshInterface {
 public:
  enum DEPTHFUNC : GLenum {
    NEVER = GL_NEVER,
    LESS = GL_LESS,
    EQUAL = GL_EQUAL,
    LESS_OR_EQUAL = GL_LEQUAL,
    GREATER = GL_GREATER,
    GREATER_OR_EQUAL = GL_GEQUAL,
    NOT_EQUAL = GL_NOTEQUAL,
    ALWAYS = GL_ALWAYS
  };
  enum RASTERMODE : GLenum { POINT = GL_POINT, LINE = GL_LINE, FILL = GL_FILL };

 protected:
  Object3D geometry{};
  std::unique_ptr<GLMaterial> material{};
  bool mesh_initialized{false};
  Mesh *cubemap_reference{};
  Camera *camera{};
  glm::mat4 modelview_matrix{};
  bool face_culling_enabled{false};
  bool depth_mask_enabled{false};
  bool is_drawn{false};
  Shader *shader_program{};
  RASTERMODE polygon_mode;

 public:
  explicit Mesh(SceneTreeNode *parent = nullptr);
  Mesh(const Object3D &obj, const GLMaterial &mat, SceneTreeNode *parent = nullptr);
  Mesh(const std::string &name, const Object3D &obj, const GLMaterial &mat, SceneTreeNode *parent = nullptr);
  Mesh(const std::string &name, const Object3D &obj, const GLMaterial &mat, Shader *shader, SceneTreeNode *parent = nullptr);
  Mesh(const std::string &name, Object3D &&obj, const GLMaterial &mat, Shader *shader, SceneTreeNode *parent = nullptr);
  virtual void bindMaterials();
  virtual void unbindMaterials();
  /**
   * @brief Computes relevant matrices , and sets up the culling + depth states
   *
   */
  void preRenderSetup() override;
  void afterRenderSetup() override;
  virtual void setupAndBind();
  virtual void bindShaders();
  virtual void releaseShaders();
  void reset() override;
  ax_no_discard bool isInitialized() const override;
  virtual void initializeGlData();
  virtual void setSceneCameraPointer(Camera *camera);
  ax_no_discard virtual const glm::mat4 &getModelViewMatrix() const { return modelview_matrix; }
  void setPolygonDrawMode(RASTERMODE mode);
  void cullBackFace();
  void cullFrontFace();
  void cullFrontAndBackFace();
  void setFaceCulling(bool value);
  void setDepthMask(bool value);
  void setDepthFunc(DEPTHFUNC func);
  Shader *getShader() { return shader_program; }
  void setShader(Shader *shader);
  ax_no_discard MaterialInterface *getMaterial() const override { return material.get(); }
  ax_no_discard const std::string &getMeshName() const override { return name; }
  void setMeshName(const std::string &new_name) override { name = new_name; }
  ax_no_discard const Object3D &getGeometry() const override { return geometry; }
  void setGeometry(const Object3D &_geometry) override { geometry = _geometry; }
  void setCubemapPointer(Mesh *cubemap_pointer) { cubemap_reference = cubemap_pointer; }
  void setDrawState(bool draw) override { is_drawn = draw; }
  ax_no_discard bool isDrawn() const override { return is_drawn; }
};

/*****************************************************************************************************************/

class CubeMesh : public Mesh {
 private:
  std::vector<float> vertices = {-1, -1, -1,  // 0
                                 1,  -1, -1,  // 1
                                 -1, 1,  -1,  // 2
                                 1,  1,  -1,  // 3
                                 -1, -1, 1,   // 4
                                 1,  -1, 1,   // 5
                                 -1, 1,  1,   // 6
                                 1,  1,  1};  // 7

  std::vector<unsigned int> indices = {0, 1, 2,  // Front face
                                       1, 3, 2,  //
                                       5, 4, 6,  // Back face
                                       6, 7, 5,  //
                                       0, 2, 6,  // Left face
                                       0, 6, 4,  //
                                       1, 5, 7,  // Right face
                                       7, 3, 1,  //
                                       3, 7, 6,  // Up face
                                       2, 3, 6,  //
                                       0, 4, 5,  // Down face
                                       0, 5, 1};

  std::vector<float> textures = {0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1};

  std::vector<float> colors = {1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0};

 public:
  explicit CubeMesh(SceneTreeNode *parent = nullptr);
  void preRenderSetup() override;
};

/*****************************************************************************************************************/

/**
 * @brief Cubemap Mesh class
 */
class CubeMapMesh : public CubeMesh {
 public:
  explicit CubeMapMesh(SceneTreeNode *parent = nullptr);
  void preRenderSetup() override;
  glm::mat4 computeFinalTransformation() override;
};

/*****************************************************************************************************************/
class QuadMesh : public Mesh {
 private:
  std::vector<float> vertices = {-1.0f, -1.0f, 0.f, -1.0f, 1.0f, 0.f, 1.0f, 1.0f, 0.f, 1.0f, -1.0f, 0.f};
  std::vector<unsigned> indices = {2, 1, 0, 3, 2, 0};
  std::vector<float> textures = {0, 0, 0, 1, 1, 1, 1, 0};

 public:
  explicit QuadMesh(SceneTreeNode *parent = nullptr);
  void preRenderSetup() override;
};

/*****************************************************************************************************************/

/**
 * @class FrameBufferMesh
 * @brief Mesh with an FBO or/and RBO attached to it
 *
 */
class FrameBufferMesh : public QuadMesh {
 public:
  FrameBufferMesh();
  FrameBufferMesh(int database_texture_index, Shader *shader);
  void preRenderSetup() override;
};

/*****************************************************************************************************************/

class BoundingBoxMesh : public Mesh {
 protected:
  geometry::BoundingBox bounding_box;
  std::vector<float> vertices;
  std::vector<unsigned> indices;

 public:
  explicit BoundingBoxMesh(SceneTreeNode *parent = nullptr);
  BoundingBoxMesh(Mesh *bound_mesh, Shader *display_shader);
  /**
   * @brief Construct a new Bounding Box Mesh using pre-computed bounding boxes
   * @param bound_mesh The Mesh we want to wrap in an aabb . Note that this mesh is used as parent in the scene graph
   * @param bounding_box The pre-computed bounding box
   * @param display_shader Shader used to display the bounding box
   */
  BoundingBoxMesh(Mesh *bound_mesh, const geometry::BoundingBox &bounding_box, Shader *display_shader);
  void afterRenderSetup() override;
  void preRenderSetup() override;
  virtual geometry::BoundingBox getBoundingBoxObject() { return bounding_box; }
};

#endif
