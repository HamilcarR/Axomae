#ifndef NOVA_SCENE_H
#define NOVA_SCENE_H
#include "aggregate/acceleration_interface.h"
#include "aggregate/device_acceleration_interface.h"
#include "camera/nova_camera.h"
#include "material/nova_material.h"
#include "primitive/nova_primitive.h"
#include "shape/nova_shape.h"
#include "texturing/nova_texturing.h"
#include <memory>

namespace nova::scene {
  struct SceneTransformations {
    /* Primary scene rotation*/
    glm::mat4 R;
    glm::mat4 inv_R;
    /* Primary scene translation*/
    glm::mat4 T;
    glm::mat4 inv_T;
    /* Primary scene transformation  (R x T)*/
    glm::mat4 M;
    glm::mat4 inv_M;
  };

  struct SceneResourcesHolder {
    texturing::TextureResourcesHolder textures_data{};
    material::MaterialResourcesHolder materials_data{};
    camera::CameraResourcesHolder camera_data{};
    primitive::PrimitivesResourcesHolder primitive_data{};
    shape::ShapeResourcesHolder shape_data{};
    aggregate::DefaultAccelerator api_accelerator{};
    std::unique_ptr<aggregate::DeviceAcceleratorInterface> device_accelerator;
    SceneTransformations scene_transformations{};
  };

}  // namespace nova::scene
#endif  // NOVA_SCENE_H
