#ifndef SCENE_H
#define SCENE_H

#include "Mesh.h"
#include "ResourceDatabaseManager.h"
#include "SceneHierarchy.h"
#include "shape/Box.h"
#include <map>
/**
 * @file Scene.h
 * @brief File implementing classes and functions relative to how the scene is represented and how to manage it
 *
 */

// TODO: [AX-34] Add object lookup system

class Camera;
class Drawable;
class Shader;
class LightingDatabase;

/**
 * @class This class manages all drawable meshes inside the scene and provides methods to sort them by types.
 */
class Scene : public EventInterface {
 public:
  struct AABB {
    nova::shape::Box aabb;
    Drawable *drawable;
  };

 private:
  std::map<float, Drawable *> sorted_transparent_meshes; /*<Sorted collection of transparent meshes , by distance to the camera*/
  std::map<float, AABB> sorted_meshes;                   /*<All meshes sorted by distance to the camera*/
  std::vector<AABB> scene;                               /*<Meshes of the scene , and their computed bounding boxes*/
  SceneTree scene_tree;                                  /*<Scene tree containing a hierarchy of transformations*/
  std::vector<Drawable *> bounding_boxes_array;          /*<Array of bounding boxes*/
  LightingDatabase *light_database{};                    /*<Light database pointer*/
  ResourceDatabaseManager &resource_manager;
  Camera *scene_camera{}; /*<Pointer on the scene camera*/
  CubeMapMesh *scene_skybox;
  bool display_bbox;
  std::vector<std::unique_ptr<Drawable>> drawable_collection;
  std::vector<Mesh *> mesh_collection;

 public:
  explicit Scene(ResourceDatabaseManager &manager);
  void setScene(const SceneTree &scene_tree, const std::vector<Mesh *> &mesh_list);
  virtual std::vector<Drawable *> getOpaqueElements() const;
  void initialize();
  void ignoreSkyboxTransformation(bool ignore);

  /**
   * @brief This method returns a vector of transparent meshes sorted in reverse order based on their
   * distance from the camera.
   */
  virtual std::vector<Drawable *> getSortedTransparentElements();
  /**
   * @brief This method returns an array of bounding boxes of each mesh in the scene.
   * Note that special meshes like the screen framebuffer , the cubemap , light sprites are not concerned by this
   * method.
   */
  virtual std::vector<Drawable *> getBoundingBoxElements();
  /**
   * @brief This method will add bounding boxes meshes to the scene.
   * @param box_shader Shader responsible for displaying bounding boxes
   */
  virtual void generateBoundingBoxes(Shader *box_shader);
  void clear();
  bool isReady();
  /**
   * @brief Set up the scene camera , and prep drawables for the next draw.
   * @param camera Pointer on the scene camera
   */
  void prepare_draw(Camera *camera);
  /**
   * @brief This method sorts the scene according to the meshes transparency ... opaque objects are first , and
   * transparent objects are sorted according to distance.
   */
  ax_no_discard std::vector<Drawable *> getSortedSceneByTransparency();
  void drawForwardTransparencyMode();
  /**
   * @brief This method draws bounding boxes on the scene meshes.
   * Note : must be used after every other mesh has been drawn , except the screen framebuffer ,
   * as BoundingBoxMesh uses the bound mesh's matrices for it's own transformations , and so , needs the updated
   * transformation matrices.
   */
  void drawBoundingBoxes();
  void setLightDatabasePointer(LightingDatabase *database) { light_database = database; }
  /**
   * @brief Set the pointer on the camera used for the next render pass
   */
  void setCameraPointer(Camera *_scene_camera);
  void updateTree();
  void setPolygonFill();
  void setPolygonPoint();
  void setPolygonWireframe();
  void displayBoundingBoxes(bool display) { display_bbox = display; }
  ax_no_discard std::vector<datastructure::NodeInterface *> getNodeByName(const std::string &name);
  ax_no_discard const CubeMapMesh &getConstSkybox() const { return *scene_skybox; }
  ax_no_discard CubeMapMesh &getSkybox() const { return *scene_skybox; }
  ax_no_discard const SceneTree &getConstSceneTreeRef() const { return scene_tree; }
  ax_no_discard SceneTree &getSceneTreeRef() { return scene_tree; }
  ax_no_discard const std::vector<Mesh *> &getMeshCollection() const { return mesh_collection; }
  /**
   * Returns the addresses of all meshes except the skybox
   */
  ax_no_discard std::vector<Mesh *> getMeshCollectionPtr() const;
  ax_no_discard std::vector<Drawable *> getDrawables() const;
  void switchEnvmap(int cubemap_id, int irradiance_id, int prefiltered_id, int lut_id);
  void processEvent(const controller::event::Event *event) override;
  void focusOnRenderable(int viewport_mouse_coord_x, int viewport_mouse_coord_y);

 private:
  /**
   * @brief Sort meshes by distance to the camera : Will erase sorted_meshes and sorted_transparent_meshes beforehand
   */
  void sortMeshesByDistanceToCamera();
};

#endif