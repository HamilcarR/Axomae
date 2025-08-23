#ifndef API_MATERIAL_H
#define API_MATERIAL_H
#include "api_common.h"

namespace nova {

  class Material {
   public:
    virtual ~Material() = default;
    /**
     * @brief Register the albedo (base color) texture for this material.
     * @param texture Unique pointer to the albedo texture.
     * @return SUCCESS if successful.
     */
    virtual ERROR_STATE registerAlbedo(TexturePtr texture) = 0;

    /**
     * @brief Register the normal map texture for this material.
     * @param texture Unique pointer to the normal texture.
     * @return SUCCESS if successful.
     */
    virtual ERROR_STATE registerNormal(TexturePtr texture) = 0;

    /**
     * @brief Register the metallic texture for this material.
     * @param texture Unique pointer to the metallic texture.
     * @return SUCCESS if successful.
     */
    virtual ERROR_STATE registerMetallic(TexturePtr texture) = 0;

    /**
     * @brief Register the emissive texture for this material.
     * @param texture Unique pointer to the emissive texture.
     * @return SUCCESS if successful.
     */
    virtual ERROR_STATE registerEmissive(TexturePtr texture) = 0;

    /**
     * @brief Register the roughness texture for this material.
     * @param texture Unique pointer to the roughness texture.
     * @return SUCCESS if successful.
     */
    virtual ERROR_STATE registerRoughness(TexturePtr texture) = 0;

    /**
     * @brief Register the opacity (alpha) texture for this material.
     * @param texture Unique pointer to the opacity texture.
     * @return SUCCESS if successful.
     */
    virtual ERROR_STATE registerOpacity(TexturePtr texture) = 0;

    /**
     * @brief Register the specular texture for this material.
     * @param texture Unique pointer to the specular texture.
     * @return SUCCESS if successful.
     */
    virtual ERROR_STATE registerSpecular(TexturePtr texture) = 0;

    /**
     * @brief Register the ambient occlusion texture for this material.
     * @param texture Unique pointer to the ambient occlusion texture.
     * @return SUCCESS if successful.
     */
    virtual ERROR_STATE registerAmbientOcclusion(TexturePtr texture) = 0;

    /**
     * @brief Set the refraction coefficient (index of refraction) for this material.
     * @param eta The refraction coefficient.
     * @return SUCCESS if successful.
     */
    virtual ERROR_STATE setRefractCoeff(float eta) = 0;

    /**
     * @brief Set the reflect fuzziness for this material.
     * @param fuzz The reflect fuzziness value.
     * @return SUCCESS if successful.
     */
    virtual ERROR_STATE setReflectFuzz(float fuzz) = 0;

    virtual Texture *getAlbedo() = 0;
    virtual const Texture *getAlbedo() const = 0;
    virtual Texture *getNormal() = 0;
    virtual const Texture *getNormal() const = 0;
    virtual Texture *getMetallic() = 0;
    virtual const Texture *getMetallic() const = 0;
    virtual Texture *getEmissive() = 0;
    virtual const Texture *getEmissive() const = 0;
    virtual Texture *getRoughness() = 0;
    virtual const Texture *getRoughness() const = 0;
    virtual Texture *getOpacity() = 0;
    virtual const Texture *getOpacity() const = 0;
    virtual Texture *getSpecular() = 0;
    virtual const Texture *getSpecular() const = 0;
    virtual Texture *getAmbientOcclusion() = 0;
    virtual const Texture *getAmbientOcclusion() const = 0;

    virtual float getRefractCoeff() const = 0;
    virtual float getReflectFuzz() const = 0;
  };

  MaterialPtr create_material();

}  // namespace nova

#endif
