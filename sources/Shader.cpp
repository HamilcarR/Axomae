#include "../includes/Shader.h"
#include "../includes/UniformNames.h"
#include <QMatrix4x4>
#include <cstring>
#include <stdexcept>

#define SHADER_ERROR_LOG_SIZE 512

/**
 * @file Shader.cpp
 * Implements functions and methods relative to the shading
 *
 */

static int success;
static char infoLog[SHADER_ERROR_LOG_SIZE];

constexpr const char *RUNTIME_ERROR_NEGATIVE_TEXTURE_UNIT = "Texture unit provided is negative";

/**
 * This function checks for compilation errors in a shader and prints an error message if there are
 * any.
 *
 * @param shader_id The ID of the shader object that needs to be checked for compilation errors.
 */
inline void shaderCompilationErrorCheck(unsigned int shader_id) {
  success = 0;
  memset(infoLog, 0, SHADER_ERROR_LOG_SIZE);
  glGetShaderiv(shader_id, GL_COMPILE_STATUS, &success);
  if (!success) {
    glGetShaderInfoLog(shader_id, SHADER_ERROR_LOG_SIZE, nullptr, infoLog);
    LOG("Shader compilation failed with error : " + std::string(infoLog), LogLevel::ERROR);
  }
}

/**
 * This function checks for errors in shader program linking and prints an error message if there is a
 * failure.
 *
 * @param program_id The ID of the shader program that needs to be checked for linking errors.
 */
inline void programLinkingErrorCheck(unsigned int program_id) {
  success = 0;
  memset(infoLog, 0, SHADER_ERROR_LOG_SIZE);
  glGetProgramiv(program_id, GL_LINK_STATUS, &success);
  if (!success) {
    glGetProgramInfoLog(program_id, SHADER_ERROR_LOG_SIZE, nullptr, infoLog);
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

Shader::Shader(const std::string vertex_code, const std::string fragment_code) : Shader() {
  type = GENERIC;
  fragment_shader_txt = fragment_code;
  vertex_shader_txt = vertex_code;
}

void Shader::enableAttributeArray(GLuint att) {
  glEnableVertexAttribArray(att);
}

void Shader::setAttributeBuffer(GLuint location, GLenum type, int offset, int tuplesize, int stride) {
  glVertexAttribPointer(location, tuplesize, type, GL_FALSE, stride, (void *)0);
}

void Shader::setTextureUniforms(std::string texture_name, Texture::TYPE type) {
  try {
    setTextureUniforms(texture_name, static_cast<int>(type));
  } catch (const std::exception &e) {
    LOG("Exception caught : " + std::string(e.what()), LogLevel::ERROR);
  }
}

void Shader::setTextureUniforms(std::string texture_name, int type) {
  if (type < 0)
    throw std::runtime_error(RUNTIME_ERROR_NEGATIVE_TEXTURE_UNIT);
  setUniform(texture_name, static_cast<int>(type));
}

void Shader::setSceneCameraPointer(Camera *camera) {
  camera_pointer = camera;
}

void Shader::updateCamera() {
  if (camera_pointer != nullptr)
    setCameraPositionUniform();
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
void Shader::setCubemapNormalMatrixUniform(glm::mat4 modelview_matrix) {
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

void Shader::setModelMatrixUniform(const glm::mat4 &matrix) {
  setUniform(uniform_name_matrix_model, matrix);
}

void Shader::setModelViewProjectionMatricesUniforms(const glm::mat4 &projection,
                                                    const glm::mat4 &view,
                                                    const glm::mat4 &model) {
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

void Shader::setUniformValue(int location, const int value) {
  glUniform1i(location, value);
}

void Shader::setUniformValue(int location, const float value) {
  glUniform1f(location, value);
}

void Shader::setUniformValue(int location, const unsigned int value) {
  glUniform1ui(location, value);
}

void Shader::setUniformValue(int location, const glm::mat4 &matrix) {
  glUniformMatrix4fv(location, 1, GL_FALSE, glm::value_ptr(matrix));
}

void Shader::setUniformValue(int location, const glm::mat3 &matrix) {
  glUniformMatrix3fv(location, 1, GL_FALSE, glm::value_ptr(matrix));
}

void Shader::setUniformValue(int location, const glm::vec4 &value) {
  glUniform4f(location, value.x, value.y, value.z, value.w);
}

void Shader::setUniformValue(int location, const glm::vec3 &value) {
  glUniform3f(location, value.x, value.y, value.z);
}

void Shader::setUniformValue(int location, const glm::vec2 &value) {
  glUniform2f(location, value.x, value.y);
}

void Shader::initializeShader() {
  if (!is_initialized) {

    errorCheck(__FILE__, __LINE__);
    const char *vertex_shader_source = (const char *)vertex_shader_txt.c_str();
    const char *fragment_shader_source = (const char *)fragment_shader_txt.c_str();
    vertex_shader = glCreateShader(GL_VERTEX_SHADER);

    glShaderSource(vertex_shader, 1, &vertex_shader_source, nullptr);
    glCompileShader(vertex_shader);

    errorCheck(__FILE__, __LINE__);
    shaderCompilationErrorCheck(vertex_shader);
    fragment_shader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragment_shader, 1, &fragment_shader_source, nullptr);
    glCompileShader(fragment_shader);
    errorCheck(__FILE__, __LINE__);
    shaderCompilationErrorCheck(fragment_shader);
    shader_program = glCreateProgram();
    glAttachShader(shader_program, vertex_shader);
    glAttachShader(shader_program, fragment_shader);

    errorCheck(__FILE__, __LINE__);
    glLinkProgram(shader_program);

    errorCheck(__FILE__, __LINE__);
    programLinkingErrorCheck(shader_program);
    errorCheck(__FILE__, __LINE__);
    is_initialized = true;
  }
}
void Shader::recompile() {
  clean();
  initializeShader();
}

void Shader::bind() {
  glUseProgram(shader_program);
}

void Shader::release() {
  glUseProgram(0);
}

void Shader::clean() {
  if (shader_program != 0) {
    LOG("Destroying shader : " + std::to_string(shader_program), LogLevel::INFO);
    glDeleteShader(vertex_shader);
    glDeleteShader(fragment_shader);
    glDeleteProgram(shader_program);
    shader_program = 0;
    is_initialized = false;
  }
}

/***********************************************************************************************************************************************************/

BlinnPhongShader::BlinnPhongShader() : Shader() {
  type = BLINN;
}

BlinnPhongShader::BlinnPhongShader(const std::string vertex_code, const std::string fragment_code)
    : Shader(vertex_code, fragment_code) {
  type = BLINN;
}

/***********************************************************************************************************************************************************/

CubemapShader::CubemapShader() : Shader() {
  type = CUBEMAP;
}

CubemapShader::CubemapShader(const std::string vertex, const std::string frag) : Shader(vertex, frag) {
  type = CUBEMAP;
}

/***********************************************************************************************************************************************************/
PBRShader::PBRShader() {
  type = PBR;
}
PBRShader::PBRShader(const std::string vertex, const std::string frag) : Shader(vertex, frag) {
  type = PBR;
}

/***********************************************************************************************************************************************************/
ScreenFramebufferShader::ScreenFramebufferShader() : Shader() {
  type = SCREEN_FRAMEBUFFER;
  post_p_blurr = false;
  post_p_sharpen = false;
  post_p_edge = false;
}

ScreenFramebufferShader::ScreenFramebufferShader(const std::string vertex, const std::string frag)
    : Shader(vertex, frag) {
  type = SCREEN_FRAMEBUFFER;
  post_p_blurr = false;
  post_p_sharpen = false;
  post_p_edge = false;
}

void ScreenFramebufferShader::setPostProcessUniforms() {
  setUniform(uniform_name_bool_blurr, post_p_blurr);
  setUniform(uniform_name_bool_edge, post_p_edge);
  setUniform(uniform_name_bool_sharpen, post_p_sharpen);
}

void ScreenFramebufferShader::setPostProcess(POST_PROCESS_TYPE type) {
  switch (type) {
    case EDGE:
      post_p_edge = true;
      post_p_sharpen = false;
      post_p_blurr = false;
      break;
    case SHARPEN:
      post_p_edge = true;
      post_p_sharpen = true;
      post_p_blurr = false;
      break;
    case BLURR:
      post_p_edge = false;
      post_p_sharpen = false;
      post_p_blurr = true;
      break;
    default:
      post_p_edge = false;
      post_p_sharpen = false;
      post_p_blurr = false;
      break;
  }
}
/***********************************************************************************************************************************************************/

BoundingBoxShader::BoundingBoxShader() : Shader() {
  type = BOUNDING_BOX;
}

BoundingBoxShader::BoundingBoxShader(const std::string vertex, const std::string fragment) : Shader(vertex, fragment) {
  type = BOUNDING_BOX;
}

/***********************************************************************************************************************************************************/

EnvmapCubemapBakerShader::EnvmapCubemapBakerShader() : Shader() {
  type = ENVMAP_CUBEMAP_CONVERTER;
}

EnvmapCubemapBakerShader::EnvmapCubemapBakerShader(const std::string vertex, const std::string fragment)
    : Shader(vertex, fragment) {
  type = ENVMAP_CUBEMAP_CONVERTER;
}

/***********************************************************************************************************************************************************/

IrradianceCubemapBakerShader::IrradianceCubemapBakerShader() : Shader() {
  type = IRRADIANCE_CUBEMAP_COMPUTE;
}
IrradianceCubemapBakerShader::IrradianceCubemapBakerShader(const std::string vertex_code,
                                                           const std::string fragment_code)
    : Shader(vertex_code, fragment_code) {
  type = IRRADIANCE_CUBEMAP_COMPUTE;
}

/***********************************************************************************************************************************************************/

EnvmapPrefilterBakerShader::EnvmapPrefilterBakerShader() : Shader() {
  type = ENVMAP_PREFILTER;
}

EnvmapPrefilterBakerShader::EnvmapPrefilterBakerShader(const std::string vertex_code, const std::string fragment_code)
    : Shader(vertex_code, fragment_code) {
  type = ENVMAP_PREFILTER;
}

void EnvmapPrefilterBakerShader::setRoughnessValue(float roughness) {
  setUniform(uniform_name_float_cubemap_prefilter_roughness, roughness);
}

void EnvmapPrefilterBakerShader::setCubeEnvmapResolution(unsigned int resolution) {
  setUniform(uniform_name_uint_prefilter_shader_envmap_resolution, resolution);
}

void EnvmapPrefilterBakerShader::setSamplesCount(unsigned int amount) {
  setUniform(uniform_name_uint_prefilter_shader_samples_count, amount);
}

/***********************************************************************************************************************************************************/

BRDFLookupTableBakerShader::BRDFLookupTableBakerShader() : Shader() {
  type = BRDF_LUT_BAKER;
}

BRDFLookupTableBakerShader::BRDFLookupTableBakerShader(const std::string vertex_code, const std::string fragment_code)
    : Shader(vertex_code, fragment_code) {
  type = BRDF_LUT_BAKER;
}
