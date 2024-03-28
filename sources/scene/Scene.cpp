#include "Scene.h"
#include "BoundingBox.h"
#include "Camera.h"
#include "Drawable.h"
#include "INodeDatabase.h"
#include "Shader.h"
#include "ShaderDatabase.h"
#include "TextureDatabase.h"
#include "utils_3D.h"

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

void Scene::setScene(std::pair<std::vector<Mesh *>, SceneTree> &to_copy) {
  to_copy.second.pushNewRoot(scene_skybox);
  scene_tree = to_copy.second;
  to_copy.first.push_back(scene_skybox);
  for (Mesh *A : to_copy.first) {
    Scene::AABB mesh;
    A->setCubemapPointer(scene_skybox);
    auto drawable = std::make_unique<Drawable>(A);
    mesh.aabb = BoundingBox(A->getGeometry().vertices);
    mesh.drawable = drawable.get();
    scene.push_back(mesh);
    drawable_collection.push_back(std::move(drawable));
  }
}

void Scene::generateBoundingBoxes(Shader *box_shader) {
  for (Scene::AABB &scene_drawable : scene) {
    Mesh *mesh = scene_drawable.drawable->getMeshPointer();
    BoundingBoxMesh *bbox_mesh =
        database::node::store<BoundingBoxMesh>(*resource_manager.getNodeDatabase(), false, mesh, scene_drawable.aabb, box_shader).object;
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
  for (const AABB &aabb : scene) {
    Drawable *A = aabb.drawable;
    GLMaterial *mat = A->getMaterialPointer();
    if (!mat->isTransparent())
      to_return.push_back(A);
    else {
      glm::mat4 modelview_matrix = A->getMeshPointer()->getModelViewMatrix();
      glm::vec3 updated_aabb_center = aabb.aabb.computeModelViewPosition(modelview_matrix);
      float dist_to_camera = glm::length(updated_aabb_center);
      sorted_transparent_meshes[dist_to_camera] = A;
    }
  }
  for (auto it = sorted_transparent_meshes.rbegin(); it != sorted_transparent_meshes.rend(); it++)
    to_return.push_back(it->second);
  return to_return;
}

void Scene::sortTransparentElements() {
  for (auto bbox : scene) {
    GLMaterial *A = bbox.drawable->getMaterialPointer();
    if (A->isTransparent()) {
      glm::mat4 modelview_matrix = bbox.drawable->getMeshPointer()->getModelViewMatrix();
      glm::vec3 updated_aabb_center = bbox.aabb.computeModelViewPosition(modelview_matrix);
      float dist_to_camera = glm::length(updated_aabb_center);
      sorted_transparent_meshes[dist_to_camera] = bbox.drawable;
    }
  }
}

std::vector<Drawable *> Scene::getSortedTransparentElements() {
  std::vector<Drawable *> transparent_meshes;
  sorted_transparent_meshes.clear();
  sortTransparentElements();
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

std::vector<NodeInterface *> Scene::getNodeByName(const std::string &name) { return scene_tree.findByName(name); }

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