#include "Scene.h"
#include "Camera.h"
#include "Drawable.h"
#include "EventController.h"
#include "INodeDatabase.h"
#include "Logger.h"
#include "Ray.h"
#include "Shader.h"
#include "ShaderDatabase.h"
#include "TextureDatabase.h"
#include "math_camera.h"

using namespace axomae;

Scene::Scene(ResourceDatabaseManager &rdm) : resource_manager(rdm), display_bbox(false) {
  scene_skybox = database::node::store<CubeMapMesh>(*resource_manager.getNodeDatabase(), true).object;
}

void Scene::updateTree() {
  scene_tree.updateOwner();
  scene_tree.updateAccumulatedTransformations();
}

void Scene::setCameraPointer(Camera *_scene_camera) {
  scene_camera = _scene_camera;
  scene_tree.pushNewRoot(scene_camera);
}

void Scene::initialize() { scene_skybox->setShader(resource_manager.getShaderDatabase()->get(Shader::CUBEMAP)); }

inline void setUpIblData(TextureDatabase &texture_database, Mesh *mesh) {}

static void make_drawable(std::vector<std::unique_ptr<Drawable>> &scene_drawables, std::vector<Scene::AABB> &bounding_boxes, Mesh *A) {
  Scene::AABB bbox_mesh;
  auto drawable = std::make_unique<Drawable>(A);
  bbox_mesh.aabb = nova::shape::Box(A->getGeometry().vertices);
  bbox_mesh.drawable = drawable.get();
  bounding_boxes.push_back(bbox_mesh);
  scene_drawables.push_back(std::move(drawable));
}

void Scene::setScene(const SceneTree &tree, const std::vector<Mesh *> &mesh_list) {
  scene_tree = tree;
  scene_tree.pushNewRoot(scene_skybox);
  scene_skybox->setCubemapPointer(scene_skybox);
  make_drawable(drawable_collection, scene, scene_skybox);
  for (Mesh *A : mesh_list) {
    A->setCubemapPointer(scene_skybox);
    make_drawable(drawable_collection, scene, A);
  }
}

void Scene::generateBoundingBoxes(Shader *box_shader) {
  for (Scene::AABB &scene_drawable : scene) {
    Mesh *mesh = scene_drawable.drawable->getMeshPointer();
    BoundingBoxMesh *bbox_mesh =
        database::node::store<BoundingBoxMesh>(*resource_manager.getNodeDatabase(), false, mesh, scene_drawable.aabb.computeAABB(), box_shader).object;
    auto bbox_drawable = std::make_unique<Drawable>(bbox_mesh);
    bounding_boxes_array.push_back(bbox_drawable.get());
    drawable_collection.push_back(std::move(bbox_drawable));
  }
}

std::vector<Drawable *> Scene::getOpaqueElements() const {
  std::vector<Drawable *> to_return;
  for (auto &aabb : scene) {
    Drawable *A = aabb.drawable;
    GLMaterial *mat = A->getMaterialPointer();
    if (!mat->isTransparent())
      to_return.push_back(A);
  }
  return to_return;
}

std::vector<Drawable *> Scene::getSortedSceneByTransparency() {
  std::vector<Drawable *> to_return;
  sorted_transparent_meshes.clear();
  sorted_meshes.clear();
  for (const AABB &aabb : scene) {
    Drawable *A = aabb.drawable;
    GLMaterial *mat = A->getMaterialPointer();
    glm::mat4 modelview_matrix = A->getMeshPointer()->getModelViewMatrix();
    glm::vec3 updated_aabb_center = modelview_matrix * glm::vec4(aabb.aabb.getPosition(), 1.f);
    float dist_to_camera = glm::length(updated_aabb_center);
    if (!mat->isTransparent())
      to_return.push_back(A);
    else {
      sorted_transparent_meshes[dist_to_camera] = A;
    }
    sorted_meshes[dist_to_camera] = aabb;
  }
  for (auto it = sorted_transparent_meshes.rbegin(); it != sorted_transparent_meshes.rend(); it++)
    to_return.push_back(it->second);
  return to_return;
}

void Scene::sortMeshesByDistanceToCamera() {
  sorted_transparent_meshes.clear();
  sorted_meshes.clear();
  for (auto bbox : scene) {
    GLMaterial *A = bbox.drawable->getMaterialPointer();
    glm::mat4 modelview_matrix = bbox.drawable->getMeshPointer()->getModelViewMatrix();
    glm::vec3 updated_aabb_center = modelview_matrix * glm::vec4(bbox.aabb.getPosition(), 1.f);
    float dist_to_camera = glm::length(updated_aabb_center);
    if (A->isTransparent()) {
      sorted_transparent_meshes[dist_to_camera] = bbox.drawable;
    }
    sorted_meshes[dist_to_camera] = bbox;
  }
}

std::vector<Drawable *> Scene::getSortedTransparentElements() {
  std::vector<Drawable *> transparent_meshes;
  sortMeshesByDistanceToCamera();
  for (auto it = sorted_transparent_meshes.rbegin(); it != sorted_transparent_meshes.rend(); it++)
    transparent_meshes.push_back(it->second);
  return transparent_meshes;
}

std::vector<Drawable *> Scene::getBoundingBoxElements() { return bounding_boxes_array; }

void Scene::clear() {
  for (auto &A : drawable_collection)
    A->clean();
  for (auto &A : bounding_boxes_array)
    A->clean();
  bounding_boxes_array.clear();
  drawable_collection.clear();
  scene.clear();
  scene_tree.clear();
  sorted_transparent_meshes.clear();
}

bool Scene::isReady() {
  for (auto object : scene)
    if (!object.drawable->ready())
      return false;
  return true;
}

void Scene::prepare_draw(Camera *_scene_camera) {
  for (auto aabb : scene) {
    aabb.drawable->setSceneCameraPointer(_scene_camera);
    aabb.drawable->startDraw();
  }
  for (auto A : bounding_boxes_array) {
    A->setSceneCameraPointer(_scene_camera);
    A->startDraw();
  }
}

void Scene::drawForwardTransparencyMode() {
  std::vector<Drawable *> meshes = getSortedSceneByTransparency();
  scene_camera->computeViewProjection();
  glm::mat4 view_matrix = scene_camera->getView();
  for (Drawable *A : meshes) {
    A->bind();
    light_database->updateShadersData(A->getMeshShaderPointer(), view_matrix);
    glDrawElements(GL_TRIANGLES, A->getMeshPointer()->getGeometry().indices.size(), GL_UNSIGNED_INT, 0);
    A->unbind();
  }
}

// TODO: [AX-39] Merge bounding box geometry
void Scene::drawBoundingBoxes() {
  if (display_bbox) {
    std::vector<Drawable *> bounding_boxes = getBoundingBoxElements();
    for (Drawable *A : bounding_boxes) {
      A->bind();
      glDrawElements(GL_TRIANGLES, A->getMeshPointer()->getGeometry().indices.size(), GL_UNSIGNED_INT, 0);
      A->unbind();
    }
  }
}

std::vector<datastructure::NodeInterface *> Scene::getNodeByName(const std::string &name) { return scene_tree.findByName(name); }

void Scene::setPolygonWireframe() {
  for (auto A : scene) {
    Mesh *tmp = A.drawable->getMeshPointer();
    if (tmp)
      tmp->setPolygonDrawMode(Mesh::LINE);
  }
}

void Scene::setPolygonPoint() {
  for (auto A : scene) {
    Mesh *tmp = A.drawable->getMeshPointer();
    if (tmp)
      tmp->setPolygonDrawMode(Mesh::POINT);
  }
}

void Scene::setPolygonFill() {
  for (auto A : scene) {
    Mesh *tmp = A.drawable->getMeshPointer();
    if (tmp)
      tmp->setPolygonDrawMode(Mesh::FILL);
  }
}

std::vector<Mesh *> Scene::getMeshCollectionPtr() const {
  std::vector<Mesh *> to_ret;
  to_ret.reserve(scene.size());
  for (auto &elem : scene)
    to_ret.push_back(elem.drawable->getMeshPointer());
  return to_ret;
}

void Scene::switchEnvmap(int cubemap_id, int irradiance_id, int prefiltered_id, int /*lut_id*/) {
  for (auto &elem : scene) {
    Mesh *mesh = elem.drawable->getMeshPointer();
    auto *material = dynamic_cast<GLMaterial *>(mesh->getMaterial());
    TextureGroup &texgroup = material->getTextureGroupRef();
    if (mesh != scene_skybox) {
      texgroup.removeTexture(Texture::IRRADIANCE);
      texgroup.addTexture(irradiance_id);
      texgroup.removeTexture(Texture::CUBEMAP);
      texgroup.addTexture(prefiltered_id);
    } else {
      texgroup.removeTexture(Texture::CUBEMAP);
      texgroup.addTexture(cubemap_id);
    }
  }
}

void Scene::focusOnRenderable(int x, int y) {
  if (!scene_camera)
    return;
  const Dim2 *dimensions = scene_camera->getScreenDimensions();
  AX_ASSERT(dimensions, "Camera pointer on screen Dim structure not valid.");
  const math::camera::camera_ray r = math::camera::ray(
      x, y, (int)dimensions->width, (int)dimensions->height, scene_camera->getProjection(), scene_camera->getView());

  for (const std::pair<const float, AABB> &it : sorted_meshes) {  // replace by space partition
    AABB elem = it.second;
    const glm::mat4 world_mat = elem.drawable->getMeshPointer()->computeFinalTransformation();
    const glm::mat4 inv_world_mat = glm::inverse(world_mat);
    const nova::Ray ray(inv_world_mat * glm::vec4(r.near, 1.f), inv_world_mat * glm::vec4(r.far, 0.f));
    glm::vec3 normal{0};
    float t = 0.f;

    if (elem.aabb.intersect(ray, scene_camera->getNear(), scene_camera->getFar(), normal, t)) {
      glm::vec3 pos = elem.aabb.getPosition();
      const glm::vec3 box_center_worldspace = world_mat * glm::vec4(pos, 1.f);
      scene_camera->focus(box_center_worldspace);
      return;
    }
  }
}

void Scene::processEvent(const controller::event::Event *event) {
  using Event = controller::event::Event;
  if (event->flag & Event::EVENT_MOUSE_L_DOUBLE) {
    focusOnRenderable(event->mouse_state.pos_x, event->mouse_state.pos_y);
  }
}
