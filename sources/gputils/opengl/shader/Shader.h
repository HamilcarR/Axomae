#ifndef SHADER_H
#define SHADER_H

#include "Texture.h"
#include "internal/common/math/utils_3D.h"
#include "internal/device/rendering/DeviceShaderInterface.h"
#include "internal/device/rendering/opengl/DebugGL.h"
#include "internal/device/rendering/opengl/init_3D.h"
#include <string>

// TODO : implement uniform buffer objects
/**
 * @file Shader.h
 */

class Camera;

class Shader : public DeviceShaderInterface {
 public:
  enum TYPE : signed {
    EMPTY = -1,
    GENERIC = 0,                    /**<Undefined shader type*/
    BLINN = 1,                      /**<Blinn-Phong shader*/
    CUBEMAP = 2,                    /**<Shader used for displaying the environment map*/
    BRDF = 3,                       /**<PBR shader type*/
    SCREEN_FRAMEBUFFER = 4,         /**<Used for post processing*/
    BOUNDING_BOX = 5,               /**<Shader used for displaying bounding boxes of meshes*/
    ENVMAP_CUBEMAP_CONVERTER = 6,   /**<Shader used to bake an equirectangular environment map to a cubemap*/
    IRRADIANCE_CUBEMAP_COMPUTE = 7, /**<Shader used to compute the irradiance map*/
    ENVMAP_PREFILTER = 8,           /**<Shader used to compute the prefiltering pass of the environment map*/
    BRDF_LUT_BAKER = 9              /**<Shader used to bake a BRDF lookup table*/
  };

 protected:
  TYPE type;
  unsigned int shader_program;
  unsigned int fragment_shader;
  unsigned int vertex_shader;
  std::string fragment_shader_txt;
  std::string vertex_shader_txt;
  Camera *camera_pointer;
  bool is_initialized;

 protected:
  Shader();
  Shader(const std::string &vertex_code, const std::string &fragment_code);

 public:
  ~Shader() override = default;
  void setType(Shader::TYPE _type) { type = _type; }
  virtual Shader::TYPE getType() { return type; }
  static Shader::TYPE getType_static() { return GENERIC; }
  void initializeShader() override;
  void recompile() override;
  void bind() override;
  void release() override;
  void clean() override;

  void setSceneCameraPointer(Camera *camera);
  virtual void updateCamera();
  void setCameraPositionUniform();
  void setCubemapNormalMatrixUniform(const glm::mat4 &cubemap_model_matrix);
  void setAllMatricesUniforms(const glm::mat4 &model);
  void setAllMatricesUniforms(const glm::mat4 &projection, const glm::mat4 &view, const glm::mat4 &model);
  void setNormalMatrixUniform(const glm::mat4 &model);
  void setInverseModelMatrixUniform(const glm::mat4 &model);
  void setInverseModelViewMatrixUniform(const glm::mat4 &view, const glm::mat4 &model);
  void setModelMatrixUniform(const glm::mat4 &model);
  void setModelViewProjection(const glm::mat4 &model);
  void setModelViewProjection(const glm::mat4 &projection, const glm::mat4 &view, const glm::mat4 &model);
  void enableAttributeArray(GLuint att);
  void setAttributeBuffer(GLuint location, GLenum type, int offset, int tuplesize, int stride = 0);
  void setShadersRawText(const std::string &vs, const std::string &fs) override;
  template<typename T>
  void setUniform(const char *name, const T &value);
  template<typename T>
  void setUniform(const std::string &name, const T &value);
  [[nodiscard]] bool isInitialized() const override { return shader_program != 0; }
  virtual void setTextureUniforms(const std::string &texture_name, GenericTexture::TYPE texture_type);
  virtual void setTextureUniforms(const std::string &texture_name, int location);

 protected:
  void setUniformValue(int location, int value);
  void setUniformValue(int location, float value);
  void setUniformValue(int location, unsigned int value);
  void setUniformValue(int location, const glm::mat4 &value);
  void setUniformValue(int location, const glm::mat3 &value);
  void setUniformValue(int location, const glm::vec4 &value);
  void setUniformValue(int location, const glm::vec3 &value);
  void setUniformValue(int location, const glm::vec2 &value);

 private:
  void setModelViewProjectionMatricesUniforms(const glm::mat4 &projection, const glm::mat4 &view, const glm::mat4 &model);
};

template<typename T>
void Shader::setUniform(const char *name, const T &value) {
  int location = glGetUniformLocation(shader_program, name);
  errorCheck(__FILE__, __func__, __LINE__);
  setUniformValue(location, value);
}

template<typename T>
void Shader::setUniform(const std::string &name, const T &value) {
  setUniform(name.c_str(), value);
}
/***********************************************************************************************************************************************************/
class BlinnPhongShader : public Shader {
 protected:
  BlinnPhongShader();
  BlinnPhongShader(std::string vertex_shader, std::string fragment_shader);

 public:
  static Shader::TYPE getType_static() { return BLINN; }
};

/***********************************************************************************************************************************************************/

class CubemapShader : public Shader {
 protected:
  CubemapShader();
  CubemapShader(std::string vertex_shader, std::string fragment_shader);

 public:
  static Shader::TYPE getType_static() { return CUBEMAP; }
};

/***********************************************************************************************************************************************************/
class BRDFShader : public Shader {
 protected:
  BRDFShader();
  BRDFShader(std::string vertex_shader, std::string fragment_shader);

 public:
  static Shader::TYPE getType_static() { return BRDF; }
};

/***********************************************************************************************************************************************************/

class ScreenFramebufferShader : public Shader {
 public:
  enum POST_PROCESS_TYPE : signed { DEFAULT = 0, EDGE = 1 << 0, SHARPEN = 1 << 1, BLURR = 1 << 2 };

 protected:
  POST_PROCESS_TYPE p_process_flag;

 protected:
  ScreenFramebufferShader();
  ScreenFramebufferShader(std::string vertex_shader, std::string fragment_shader);

 public:
  void setPostProcess(POST_PROCESS_TYPE postp);
  void setPostProcessUniforms();
  static Shader::TYPE getType_static() { return SCREEN_FRAMEBUFFER; }
};

/***********************************************************************************************************************************************************/

class BoundingBoxShader : public Shader {
 protected:
  BoundingBoxShader();
  BoundingBoxShader(std::string vertex_shader, std::string fragment_shader);

 public:
  static Shader::TYPE getType_static() { return BOUNDING_BOX; }
};

/***********************************************************************************************************************************************************/

class EnvmapCubemapBakerShader : public Shader {
 protected:
  EnvmapCubemapBakerShader();
  EnvmapCubemapBakerShader(std::string vertex_shader, std::string fragment_shader);

 public:
  static Shader::TYPE getType_static() { return ENVMAP_CUBEMAP_CONVERTER; }
};

/***********************************************************************************************************************************************************/

class IrradianceCubemapBakerShader : public Shader {
 protected:
  IrradianceCubemapBakerShader();
  IrradianceCubemapBakerShader(std::string vertex_shader, std::string fragment_shader);

 public:
  static Shader::TYPE getType_static() { return IRRADIANCE_CUBEMAP_COMPUTE; }
};

/***********************************************************************************************************************************************************/

class EnvmapPrefilterBakerShader : public Shader {
 protected:
  EnvmapPrefilterBakerShader();
  EnvmapPrefilterBakerShader(std::string vertex_shader, std::string fragment_shader);

 public:
  virtual void setRoughnessValue(float roughness);
  virtual void setCubeEnvmapResolution(unsigned int resolution);
  virtual void setSamplesCount(unsigned sample_count);
  static Shader::TYPE getType_static() { return ENVMAP_PREFILTER; }
};

/***********************************************************************************************************************************************************/

class BRDFLookupTableBakerShader : public Shader {
 protected:
  BRDFLookupTableBakerShader();
  BRDFLookupTableBakerShader(std::string vertex_shader, std::string fragment_shader);

 public:
  static Shader::TYPE getType_static() { return BRDF_LUT_BAKER; }
};

/***********************************************************************************************************************************************************/

#endif
