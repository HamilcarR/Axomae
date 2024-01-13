#ifndef MATERIAL_H
#define MATERIAL_H

#include "Shader.h"
#include "TextureGroup.h"

// TODO: [AX-7] Add polymorphic material hierarchy

/**
 * @file Material.h
 * Material implementation
 *
 */

/**
 * @brief Material class implementation
 *
 */
class Material {
 public:
  /**
   * @brief Enumeration mapping OpenGL values for blending constants
   *
   */
  enum BLENDFUNC : unsigned {
    ZERO = GL_ZERO,
    ONE = GL_ONE,
    SOURCE_ALPHA = GL_SRC_ALPHA,
    ONE_MINUS_SOURCE_ALPHA = GL_ONE_MINUS_SRC_ALPHA,
    DEST_ALPHA = GL_DST_ALPHA,
    ONE_MINUS_DEST_ALPHA = GL_ONE_MINUS_DST_ALPHA,
    CSTE_COLOR = GL_CONSTANT_COLOR,
    ONE_MINUS_CSTE_COLOR = GL_ONE_MINUS_CONSTANT_COLOR,
    CSTE_ALPHA = GL_CONSTANT_ALPHA,
    ONE_MINUS_CSTE_ALPHA = GL_ONE_MINUS_CONSTANT_ALPHA
  };

  /**
   * @brief Construct a new Material object
   *
   */
  Material();

  /**
   * @brief Add a texture from the texture database to the material structure
   *
   * @param texture_database_index Index of the texture in the database map
   * @see Texture::TYPE
   */
  void addTexture(int texture_database_index);

  /**
   * @brief Set the Emissive Factor object
   *
   * @param factor New emissive factor
   */
  virtual void setEmissiveFactor(float factor) { emissive_factor = factor; }

  /**
   * @brief Bind the material data to the shader
   *
   */
  virtual void bind();

  /**
   * @brief Unbind the material data
   *
   */
  virtual void unbind();

  /**
   * @brief Cleans the material data
   *
   */
  virtual void clean();

  /**
   * @brief Initialize materials and textures
   *
   */
  virtual void initializeMaterial();

  /**
   * @brief Sets up the shader pointer used for the mesh
   *
   */
  virtual void setShaderPointer(Shader *shader) { shader_program = shader; }

  /**
   * @brief Set the Refractive Index
   *
   * @param n1 Index of the first medium
   * @param n2 Index of the second medium
   */
  virtual void setRefractiveIndexValue(float n1, float n2);

  /**
   * @brief Enable transparency for the material
   *
   */
  virtual void enableBlend();

  /**
   * @brief Disable transparency for this material
   *
   */
  virtual void disableBlend();

  /**
   * @brief Specify the blending function
   *
   * @param source_factor Source factor
   * @param dest_factor Destination factor
   */
  virtual void setBlendFunc(BLENDFUNC source_factor, BLENDFUNC dest_factor);

  /**
   * @brief Set the Transparency property
   *
   * @param transparency_value
   */
  void setTransparency(float transparency_value) { alpha_factor = 1.f - transparency_value; }

  /**
   * @brief Checks if the material has an opacity value != 0
   *
   * @return true If the alpha_factor property is < 1.f or if a used texture is an opacity texture
   * @return false If the alpha_factor == 1
   */
  virtual bool isTransparent();
  /**
   * @brief Get the Texture Group value
   *
   * @return TextureGroup
   */
  [[nodiscard]] TextureGroup getTextureGroup() const { return textures_group; }
  [[nodiscard]] TextureGroup &getTextureGroupRef() { return textures_group; }
  /**
   * @brief Get the Dielectric Factor value
   *
   * @return float
   */
  [[nodiscard]] float getDielectricFactor() const { return dielectric_factor; }

  /**
   * @brief Get the Roughness Factor value
   *
   * @return float
   */
  [[nodiscard]] float getRoughnessFactor() const { return roughness_factor; }

  /**
   * @brief Get the Transmission Factor value
   *
   * @return float
   */
  [[nodiscard]] float getTransmissionFactor() const { return transmission_factor; }

  /**
   * @brief Get the Emissive Factor value
   *
   * @return float
   */
  [[nodiscard]] float getEmissiveFactor() const { return emissive_factor; }

  /**
   * @brief Get the Alpha Factor value
   *
   * @return float
   */
  [[nodiscard]] float getAlphaFactor() const { return alpha_factor; }

  /**
   * @brief Get the Refractive Index value
   *
   * @return glm::vec2
   */
  [[nodiscard]] glm::vec2 getRefractiveIndex() const { return refractive_index; }

  /**
   * @brief Get the Shader Program object
   *
   * @return Shader*
   */
  [[nodiscard]] Shader *getShaderProgram() const { return shader_program; }

  /**
   * @brief Get the Transparency value
   *
   * @return true
   * @return false
   */
  [[nodiscard]] bool getTransparency() const { return is_transparent; }

  [[nodiscard]] bool hasTextures() const { return !textures_group.isEmpty(); }

 protected:
  TextureGroup textures_group; /**<A structure of every type of texture to be bound*/
  float dielectric_factor;     /**<Metallic factor : 0.0 = full dielectric , 1.0 = full metallic*/
  float roughness_factor;      /**<Roughness factor : 0.0 = smooth , 1.0 = rough*/
  float transmission_factor;   /**<Defines amount of light transmitted through the surface*/
  float emissive_factor;       /**<Defines the amount of light generated by the material*/
  float shininess;             /**<Defines the specular reflection strength*/
  float alpha_factor;          /**<Defines the material's opacity value*/
  glm::vec2 refractive_index;  /**<Defines the fresnel IOR*/
  Shader *shader_program;      /**<Pointer on the shader*/
  bool is_transparent;         /**<Defines if the material has transparent property*/
};

#endif
