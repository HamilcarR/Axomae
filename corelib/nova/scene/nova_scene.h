#ifndef NOVA_SCENE_H
#define NOVA_SCENE_H
#include "aggregate/nova_acceleration.h"
#include "camera/nova_camera.h"
#include "material/nova_material.h"
#include "primitive/nova_primitive.h"
#include "shape/nova_shape.h"
#include "texturing/nova_texturing.h"
#include <memory>
namespace nova::scene {
  class SceneTransformations {
   private:
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

    void setTranslation(const glm::mat4 &translation);
    void setRotation(const glm::mat4 &rotation);
    void setInvTranslation(const glm::mat4 &inv_translation);
    void setInvRotation(const glm::mat4 &inv_rotation);
    void setModel(const glm::mat4 &model);
    void setInvModel(const glm::mat4 &inv_model);
    void setPvm(const glm::mat4 &pvm);
    void setInvPvm(const glm::mat4 &inv_pvm);
    void setVm(const glm::mat4 &vm);
    void setInvVm(const glm::mat4 &inv_vm);
    void setNormalMatrix(const glm::mat3 &normal_mat);
    [[nodiscard]] const glm::mat4 &getTranslation() const;
    [[nodiscard]] const glm::mat4 &getRotation() const;
    [[nodiscard]] const glm::mat4 &getInvTranslation() const;
    [[nodiscard]] const glm::mat4 &getInvRotation() const;
    [[nodiscard]] const glm::mat4 &getModel() const;
    [[nodiscard]] const glm::mat4 &getInvModel() const;
    [[nodiscard]] const glm::mat4 &getPvm() const;
    [[nodiscard]] const glm::mat4 &getInvPvm() const;
    [[nodiscard]] const glm::mat4 &getVm() const;
    [[nodiscard]] const glm::mat4 &getInvVm() const;
    [[nodiscard]] const glm::mat3 &getNormalMatrix() const;
  };

  // TODO : use memory pool
  struct SceneResourcesHolder {
    texturing::TextureRawData envmap_data{};
    texturing::TextureResourcesHolder textures_data{};
    material::MaterialResourcesHolder materials_data{};
    camera::CameraResourcesHolder camera_data{};
    primitive::PrimitivesResourcesHolder primitive_data{};
    shape::ShapeResourcesHolder shape_data{};
    aggregate::Accelerator acceleration_data{};
    SceneTransformations scene_transformations{};
  };

}  // namespace nova::scene
#endif  // NOVA_SCENE_H
