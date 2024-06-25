#include "nova_scene.h"
using namespace nova::scene;

const glm::mat4 &SceneTransformations::getTranslation() const { return T; }

const glm::mat4 &SceneTransformations::getRotation() const { return R; }

const glm::mat4 &SceneTransformations::getInvTranslation() const { return inv_T; }

const glm::mat4 &SceneTransformations::getInvRotation() const { return inv_R; }

const glm::mat4 &SceneTransformations::getModel() const { return M; }

const glm::mat4 &SceneTransformations::getInvModel() const { return inv_M; }

const glm::mat4 &SceneTransformations::getPvm() const { return PVM; }

const glm::mat4 &SceneTransformations::getInvPvm() const { return inv_PVM; }

const glm::mat4 &SceneTransformations::getVm() const { return VM; }

const glm::mat4 &SceneTransformations::getInvVm() const { return inv_VM; }

const glm::mat3 &SceneTransformations::getNormalMatrix() const { return N; }

void SceneTransformations::setTranslation(const glm::mat4 &translation) { T = translation; }

void SceneTransformations::setRotation(const glm::mat4 &rotation) { R = rotation; }

void SceneTransformations::setInvTranslation(const glm::mat4 &inv_translation) { inv_T = inv_translation; }

void SceneTransformations::setInvRotation(const glm::mat4 &inv_rotation) { inv_R = inv_rotation; }

void SceneTransformations::setModel(const glm::mat4 &model) { M = model; }

void SceneTransformations::setInvModel(const glm::mat4 &inv_model) { inv_M = inv_model; }

void SceneTransformations::setPvm(const glm::mat4 &pvm) { PVM = pvm; }

void SceneTransformations::setInvPvm(const glm::mat4 &inv_pvm) { inv_PVM = inv_pvm; }

void SceneTransformations::setVm(const glm::mat4 &vm) { VM = vm; }

void SceneTransformations::setInvVm(const glm::mat4 &inv_vm) { inv_VM = inv_vm; }

void SceneTransformations::setNormalMatrix(const glm::mat3 &normal_mat) { N = normal_mat; }
