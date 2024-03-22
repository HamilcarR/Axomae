#include "Material.h"
#include "DebugGL.h"
#include "UniformNames.h"

GLMaterial::GLMaterial() {
  dielectric_factor = 0.f;
  shininess = 100.f;
  roughness_factor = 0.f;
  transmission_factor = 0.f;
  emissive_factor = 10.f;
  alpha_factor = 1.f;
  refractive_index = glm::vec2(1.f, 1.1f);
  shader_program = nullptr;
  is_transparent = false;
}

void GLMaterial::do_copy_constr(const GLMaterial &copy) {
  if (this != &copy) {
    textures_group = copy.textures_group;
    dielectric_factor = copy.dielectric_factor;
    roughness_factor = copy.roughness_factor;
    transmission_factor = copy.transmission_factor;
    emissive_factor = copy.emissive_factor;
    shininess = copy.shininess;
    alpha_factor = copy.alpha_factor;
    refractive_index = copy.refractive_index;
    shader_program = copy.shader_program;
    is_transparent = copy.is_transparent;
  }
}

void GLMaterial::do_move_constr(GLMaterial &&move) noexcept {
  textures_group = std::move(move.textures_group);
  dielectric_factor = move.dielectric_factor;
  roughness_factor = move.roughness_factor;
  transmission_factor = move.transmission_factor;
  emissive_factor = move.emissive_factor;
  shininess = move.shininess;
  alpha_factor = move.alpha_factor;
  refractive_index = move.refractive_index;
  shader_program = move.shader_program;
  is_transparent = move.is_transparent;
}

GLMaterial::GLMaterial(const GLMaterial &copy) { do_copy_constr(copy); }

GLMaterial::GLMaterial(GLMaterial &&move) noexcept { do_move_constr(std::forward<GLMaterial>(move)); }

GLMaterial &GLMaterial::operator=(const GLMaterial &copy) {
  do_copy_constr(copy);
  return *this;
}

GLMaterial &GLMaterial::operator=(GLMaterial &&move) noexcept {
  do_move_constr(std::forward<GLMaterial>(move));
  return *this;
}

bool GLMaterial::isTransparent() const {
  Texture *tex_opacity = textures_group.getTexturePointer(Texture::OPACITY);
  Texture *tex_diffuse = textures_group.getTexturePointer(Texture::DIFFUSE);
  return alpha_factor < 1.f || (tex_opacity && !tex_opacity->isDummyTexture()) ||
         (tex_diffuse && dynamic_cast<DiffuseTexture *>(tex_diffuse)->hasTransparency());
}

void GLMaterial::syncTransparencyState() { is_transparent = isTransparent(); }

void GLMaterial::setRefractiveIndexValue(float n1, float n2) {
  refractive_index = glm::vec2(n1, n2);
  if (shader_program)
    shader_program->setUniform(uniform_name_vec2_material_refractive_index, refractive_index);
}

void GLMaterial::addTexture(int index) { textures_group.addTexture(index); }

void GLMaterial::bind() {
  if (is_transparent) {
    enableBlend();
    setBlendFunc(SOURCE_ALPHA, ONE_MINUS_SOURCE_ALPHA);
  }
  std::string material = std::string(uniform_name_str_material_struct_name) + std::string(".");
  shader_program->setUniform(material + uniform_name_float_material_transparency_factor, alpha_factor);
  textures_group.bind();
}

void GLMaterial::unbind() { disableBlend(); }

void GLMaterial::initializeMaterial() {
  syncTransparencyState();
  if (shader_program) {
    textures_group.initializeGlTextureData(shader_program);
    std::string material = std::string(uniform_name_str_material_struct_name) + std::string(".");
    shader_program->setUniform(material + uniform_name_vec2_material_refractive_index, refractive_index);
    shader_program->setUniform(material + uniform_name_float_material_dielectric_factor, dielectric_factor);
    shader_program->setUniform(material + uniform_name_float_material_roughness_factor, roughness_factor);
    shader_program->setUniform(material + uniform_name_float_material_transmission_factor, transmission_factor);
    shader_program->setUniform(material + uniform_name_float_material_emissive_factor, emissive_factor);
    shader_program->setUniform(material + uniform_name_float_material_shininess_factor, shininess);
    errorCheck(__FILE__, __LINE__);
  }
}

void GLMaterial::clean() { textures_group.clean(); }

void GLMaterial::enableBlend() { glEnable(GL_BLEND); }

void GLMaterial::disableBlend() { glDisable(GL_BLEND); }

void GLMaterial::setBlendFunc(BLENDFUNC source_factor, BLENDFUNC dest_factor) { glBlendFunc(source_factor, dest_factor); }

math::geometry::Vect2D GLMaterial::getRefractiveIndex() const {
  math::geometry::Vect2D ret{refractive_index.x, refractive_index.y};
  return ret;
}
