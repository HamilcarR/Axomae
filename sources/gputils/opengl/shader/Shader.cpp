#include "Shader.h"
#include "Camera.h"
#include "DebugGL.h"
#include "UniformNames.h"
#include "glsl.h"
#include "init_3D.h"
#include <cstring>
#include <stdexcept>

/**
 * @file Shader.cpp
 * Implements functions and methods relative to the shading
 */

using namespace shader_utils;

constexpr unsigned SHADER_ERROR_LOG_SIZE = 512;
static int success;
static char infoLog[SHADER_ERROR_LOG_SIZE];

constexpr const char *RUNTIME_ERROR_NEGATIVE_TEXTURE_UNIT = "Texture unit provided is negative";

inline void shaderCompilationErrorCheck(unsigned int shader_id) {
  success = 0;
  std::memset(infoLog, 0, SHADER_ERROR_LOG_SIZE);
  ax_glGetShaderiv(shader_id, GL_COMPILE_STATUS, &success);
  if (!success) {
    ax_glGetShaderInfoLog(shader_id, SHADER_ERROR_LOG_SIZE, nullptr, infoLog);
    LOG("Shader compilation failed with error : " + std::string(infoLog), LogLevel::ERROR);
  }
}

inline void programLinkingErrorCheck(unsigned int program_id) {
  success = 0;
  std::memset(infoLog, 0, SHADER_ERROR_LOG_SIZE);
  ax_glGetProgramiv(program_id, GL_LINK_STATUS, &success);
  if (!success) {
    ax_glGetProgramInfoLog(program_id, SHADER_ERROR_LOG_SIZE, nullptr, infoLog);
    LOG("Shader linkage failed with error : " + std::string(infoLog), LogLevel::ERROR);
  }
}

Shader::Shader() {
  type = GENERIC;
  is_initialized = false;
  camera_pointer = nullptr;
  is_initialized = false;
  vertex_shader_txt = "";
  fragment_shader_txt = "";
  shader_program = 0;
  fragment_shader = vertex_shader = 0;
}

Shader::Shader(const std::string &vertex_code, const std::string &fragment_code) : Shader() {
  type = GENERIC;
  fragment_shader_txt = fragment_code;
  vertex_shader_txt = vertex_code;
}

void Shader::enableAttributeArray(GLuint att) { ax_glEnableVertexAttribArray(att); }

void Shader::setAttributeBuffer(GLuint location, GLenum etype, int offset, int tuplesize, int stride) {
  ax_glVertexAttribPointer(location, tuplesize, etype, GL_FALSE, stride, (void *)0);
}

void Shader::setTextureUniforms(const std::string &texture_name, GenericTexture::TYPE texture_type) {
  try {
    setTextureUniforms(texture_name, static_cast<int>(texture_type));
  } catch (const std::exception &e) {
    LOG("Exception caught : " + std::string(e.what()), LogLevel::ERROR);
  }
}

void Shader::setTextureUniforms(const std::string &texture_name, int texture_type) {
  if (type < 0)
    throw std::runtime_error(RUNTIME_ERROR_NEGATIVE_TEXTURE_UNIT);
  setUniform(texture_name, static_cast<int>(texture_type));
}

void Shader::setSceneCameraPointer(Camera *camera) { camera_pointer = camera; }

void Shader::updateCamera() {
  if (camera_pointer != nullptr)
    setCameraPositionUniform();
}

void Shader::setShadersRawText(const std::string &vs, const std::string &fs) {
  fragment_shader_txt = fs;
  vertex_shader_txt = vs;
}

void Shader::setCameraPositionUniform() {
  if (camera_pointer)
    setUniform(uniform_name_vec3_camera_position, camera_pointer->getPosition());
}

void Shader::setAllMatricesUniforms(const glm::mat4 &model) {
  setModelViewProjection(model);
  setNormalMatrixUniform(model);
  setInverseModelMatrixUniform(model);
  if (camera_pointer != nullptr)
    setInverseModelViewMatrixUniform(camera_pointer->getView(), model);
}
void Shader::setCubemapNormalMatrixUniform(const glm::mat4 &modelview_matrix) {
  setUniform(uniform_name_cubemap_matrix_normal, glm::mat3(glm::transpose(glm::inverse(modelview_matrix))));
}

void Shader::setAllMatricesUniforms(const glm::mat4 &projection, const glm::mat4 &view, const glm::mat4 &model) {
  setModelViewProjection(projection, view, model);
  setNormalMatrixUniform(model);
}

void Shader::setInverseModelViewMatrixUniform(const glm::mat4 &view, const glm::mat4 &model) {
  glm::mat4 inverse = glm::inverse(view * model);
  setUniform(uniform_name_matrix_inverse_modelview, inverse);
}

void Shader::setNormalMatrixUniform(const glm::mat4 &model) {
  setUniform(uniform_name_matrix_normal, glm::mat3(glm::transpose(glm::inverse(camera_pointer->getView() * model))));
}

void Shader::setInverseModelMatrixUniform(const glm::mat4 &model) {
  auto inverse_model = glm::inverse(model);
  setUniform(uniform_name_matrix_inverse_model, inverse_model);
}

void Shader::setModelMatrixUniform(const glm::mat4 &matrix) { setUniform(uniform_name_matrix_model, matrix); }

void Shader::setModelViewProjectionMatricesUniforms(const glm::mat4 &projection, const glm::mat4 &view, const glm::mat4 &model) {
  glm::mat4 mvp = projection * view * model;
  glm::mat4 modelview_matrix = view * model;
  glm::mat4 view_projection = projection * view;
  setUniform(uniform_name_matrix_modelview, modelview_matrix);
  setUniform(uniform_name_matrix_model_view_projection, mvp);
  setUniform(uniform_name_matrix_view_projection, view_projection);
  setUniform(uniform_name_matrix_model, model);
  setUniform(uniform_name_matrix_view, view);
  setUniform(uniform_name_matrix_projection, projection);
}

void Shader::setModelViewProjection(const glm::mat4 &model) {
  if (camera_pointer != nullptr) {
    glm::mat4 view = camera_pointer->getView();
    glm::mat4 projection = camera_pointer->getProjection();
    updateCamera();
    setModelViewProjectionMatricesUniforms(projection, view, model);
  }
}

void Shader::setModelViewProjection(const glm::mat4 &projection, const glm::mat4 &view, const glm::mat4 &model) {
  updateCamera();
  setModelViewProjectionMatricesUniforms(projection, view, model);
}

void Shader::setUniformValue(int location, const int value) { ax_glUniform1i(location, value); }

void Shader::setUniformValue(int location, const float value) { ax_glUniform1f(location, value); }

void Shader::setUniformValue(int location, const unsigned int value) { ax_glUniform1ui(location, value); }

void Shader::setUniformValue(int location, const glm::mat4 &matrix) { ax_glUniformMatrix4fv(location, 1, GL_FALSE, glm::value_ptr(matrix)); }

void Shader::setUniformValue(int location, const glm::mat3 &matrix) { ax_glUniformMatrix3fv(location, 1, GL_FALSE, glm::value_ptr(matrix)); }

void Shader::setUniformValue(int location, const glm::vec4 &value) { ax_glUniform4f(location, value.x, value.y, value.z, value.w); }

void Shader::setUniformValue(int location, const glm::vec3 &value) { ax_glUniform3f(location, value.x, value.y, value.z); }

void Shader::setUniformValue(int location, const glm::vec2 &value) { ax_glUniform2f(location, value.x, value.y); }

void Shader::initializeShader() {
  if (!is_initialized) {
    const char *vertex_shader_source = (const char *)vertex_shader_txt.c_str();
    const char *fragment_shader_source = (const char *)fragment_shader_txt.c_str();
    vertex_shader = glCreateShader(GL_VERTEX_SHADER);
    ax_glShaderSource(vertex_shader, 1, &vertex_shader_source, nullptr);
    ax_glCompileShader(vertex_shader);
    shaderCompilationErrorCheck(vertex_shader);
    fragment_shader = glCreateShader(GL_FRAGMENT_SHADER);
    ax_glShaderSource(fragment_shader, 1, &fragment_shader_source, nullptr);
    ax_glCompileShader(fragment_shader);
    shaderCompilationErrorCheck(fragment_shader);
    shader_program = glCreateProgram();
    ax_glAttachShader(shader_program, vertex_shader);
    ax_glAttachShader(shader_program, fragment_shader);
    ax_glLinkProgram(shader_program);
    programLinkingErrorCheck(shader_program);
    is_initialized = true;
  }
}
void Shader::recompile() {
  clean();
  initializeShader();
}

void Shader::bind() { ax_glUseProgram(shader_program); }

void Shader::release() { ax_glUseProgram(0); }

void Shader::clean() {
  if (shader_program != 0) {
    LOG("Destroying shader : " + std::to_string(shader_program), LogLevel::INFO);
    ax_glDeleteShader(vertex_shader);
    ax_glDeleteShader(fragment_shader);
    ax_glDeleteProgram(shader_program);
    shader_program = 0;
    is_initialized = false;
  }
}

/***********************************************************************************************************************************************************/

BlinnPhongShader::BlinnPhongShader() : Shader(glsl_utils::phong_vert(), glsl_utils::phong_frag()) { type = BLINN; }
BlinnPhongShader::BlinnPhongShader(std::string vertex_shader, std::string fragment_shader) : Shader(vertex_shader, fragment_shader) { type = BLINN; }
/***********************************************************************************************************************************************************/

CubemapShader::CubemapShader() : Shader(glsl_utils::cubemap_vert(), glsl_utils::cubemap_frag()) { type = CUBEMAP; }
CubemapShader::CubemapShader(std::string vertex_shader, std::string fragment_shader) : Shader(vertex_shader, fragment_shader) { type = CUBEMAP; }
/***********************************************************************************************************************************************************/
BRDFShader::BRDFShader() : Shader(glsl_utils::pbr_vert(), glsl_utils::pbr_frag()) { type = BRDF; }
BRDFShader::BRDFShader(std::string vertex_shader, std::string fragment_shader) : Shader(vertex_shader, fragment_shader) { type = BRDF; }
/***********************************************************************************************************************************************************/
ScreenFramebufferShader::ScreenFramebufferShader() : Shader(glsl_utils::screen_fbo_vert(), glsl_utils::screen_fbo_frag()) {
  type = SCREEN_FRAMEBUFFER;
  p_process_flag = DEFAULT;
}
ScreenFramebufferShader::ScreenFramebufferShader(std::string vertex_shader, std::string fragment_shader) : Shader(vertex_shader, fragment_shader) {
  type = SCREEN_FRAMEBUFFER;
  p_process_flag = DEFAULT;
}

void ScreenFramebufferShader::setPostProcessUniforms() {
  setUniform(uniform_name_bool_blurr, p_process_flag == BLURR);
  setUniform(uniform_name_bool_edge, p_process_flag == EDGE);
  setUniform(uniform_name_bool_sharpen, p_process_flag == SHARPEN);
}

void ScreenFramebufferShader::setPostProcess(POST_PROCESS_TYPE type) { p_process_flag = type; }
/***********************************************************************************************************************************************************/

BoundingBoxShader::BoundingBoxShader() : Shader(glsl_utils::bbox_vert(), glsl_utils::bbox_frag()) { type = BOUNDING_BOX; }
BoundingBoxShader::BoundingBoxShader(std::string vertex_shader, std::string fragment_shader) : Shader(vertex_shader, fragment_shader) {
  type = BOUNDING_BOX;
}
/***********************************************************************************************************************************************************/

EnvmapCubemapBakerShader::EnvmapCubemapBakerShader() : Shader(glsl_utils::envmap_bake_vert(), glsl_utils::envmap_bake_frag()) {
  type = ENVMAP_CUBEMAP_CONVERTER;
}
EnvmapCubemapBakerShader::EnvmapCubemapBakerShader(std::string vertex_shader, std::string fragment_shader) : Shader(vertex_shader, fragment_shader) {
  type = ENVMAP_CUBEMAP_CONVERTER;
}

/***********************************************************************************************************************************************************/

IrradianceCubemapBakerShader::IrradianceCubemapBakerShader() : Shader(glsl_utils::envmap_bake_vert(), glsl_utils::irradiance_baker_frag()) {
  type = IRRADIANCE_CUBEMAP_COMPUTE;
}
IrradianceCubemapBakerShader::IrradianceCubemapBakerShader(std::string vertex_shader, std::string fragment_shader)
    : Shader(vertex_shader, fragment_shader) {
  type = IRRADIANCE_CUBEMAP_COMPUTE;
}
/***********************************************************************************************************************************************************/

EnvmapPrefilterBakerShader::EnvmapPrefilterBakerShader() : Shader(glsl_utils::envmap_bake_vert(), glsl_utils::envmap_prefilter_frag()) {
  type = ENVMAP_PREFILTER;
}

EnvmapPrefilterBakerShader::EnvmapPrefilterBakerShader(std::string vertex_shader, std::string fragment_shader)
    : Shader(vertex_shader, fragment_shader) {
  type = ENVMAP_PREFILTER;
}

void EnvmapPrefilterBakerShader::setRoughnessValue(float roughness) { setUniform(uniform_name_float_cubemap_prefilter_roughness, roughness); }

void EnvmapPrefilterBakerShader::setCubeEnvmapResolution(unsigned int resolution) {
  setUniform(uniform_name_uint_prefilter_shader_envmap_resolution, resolution);
}

void EnvmapPrefilterBakerShader::setSamplesCount(unsigned int amount) { setUniform(uniform_name_uint_prefilter_shader_samples_count, amount); }

/***********************************************************************************************************************************************************/

BRDFLookupTableBakerShader::BRDFLookupTableBakerShader() : Shader(glsl_utils::envmap_bake_vert(), glsl_utils::brdf_lut_frag()) {
  type = BRDF_LUT_BAKER;
}
BRDFLookupTableBakerShader::BRDFLookupTableBakerShader(std::string vertex_shader, std::string fragment_shader)
    : Shader(vertex_shader, fragment_shader) {
  type = BRDF_LUT_BAKER;
}
