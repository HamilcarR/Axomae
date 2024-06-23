
#ifndef NOVA_SCENE_H
#define NOVA_SCENE_H
#include "aggregate/nova_acceleration.h"
#include "material/nova_material.h"
#include "primitive/nova_primitive.h"
#include "shape/nova_shape.h"
#include "texturing/nova_texturing.h"
#include <memory>
namespace nova::scene {

  // TODO : use memory pool
  class SceneResourcesHolder {
   public:
    texturing::TextureRawData envmap_data{};
    texturing::TextureResourcesHolder textures_data{};
    material::MaterialResourcesHolder materials_data{};
    camera::CameraResourcesHolder camera_data{};
    primitive::PrimitivesResourcesHolder primitive_data{};
    shape::ShapeResourceHolder shape_data{};
    aggregate::Accelerator acceleration_data{};
  };

}  // namespace nova::scene
#endif  // NOVA_SCENE_H
