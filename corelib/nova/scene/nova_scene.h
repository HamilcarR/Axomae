#ifndef NOVA_SCENE_H
#define NOVA_SCENE_H
#include "aggregate/acceleration_interface.h"
#include "camera/nova_camera.h"
#include "material/nova_material.h"
#include "primitive/nova_primitive.h"
#include "shape/nova_shape.h"
#include "texturing/nova_texturing.h"
#include <memory>
namespace nova::scene {
  class SceneTransformations {
   public:
    /* View x Scene Transfo */
    glm::mat4 VM;
    glm::mat4 inv_VM;
    /* Projection * View * Scene Transfo*/
    glm::mat4 PVM;
    glm::mat4 inv_PVM;
    /* Primary scene rotation*/
    glm::mat4 R;
    glm::mat4 inv_R;
    /* Primary scene translation*/
    glm::mat4 T;
    glm::mat4 inv_T;
    /* Primary scene transformation  (R x T)*/
    glm::mat4 M;
    glm::mat4 inv_M;
    /* Normal matrix */
    glm::mat3 N;

   public:
    CLASS_CM(SceneTransformations)
  };

  // TODO : use memory pool
  struct SceneResourcesHolder {
    texturing::TextureRawData envmap_data{};
    texturing::TextureResourcesHolder textures_data{};
    material::MaterialResourcesHolder materials_data{};
    camera::CameraResourcesHolder camera_data{};
    primitive::PrimitivesResourcesHolder primitive_data{};
    shape::ShapeResourcesHolder shape_data{};
    aggregate::DefaultAccelerator api_accelerator{};
    SceneTransformations scene_transformations{};
  };

}  // namespace nova::scene
#endif  // NOVA_SCENE_H
