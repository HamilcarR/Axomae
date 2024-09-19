#ifndef TEXTUREPACKSAMPLER_H
#define TEXTUREPACKSAMPLER_H

#include "texturing/nova_texturing.h"
#include <internal/macro/project_macros.h>

namespace nova::material {

  struct texture_pack {
    texturing::NovaTextureInterface albedo;
    texturing::NovaTextureInterface metallic;
    texturing::NovaTextureInterface roughness;
    texturing::NovaTextureInterface ao;
    texturing::NovaTextureInterface normalmap;
    texturing::NovaTextureInterface emissive;
    texturing::NovaTextureInterface specular;
    texturing::NovaTextureInterface opacity;
  };

  class TexturePackSampler {
   private:
    texture_pack tpack{};

   public:
    CLASS_DCM(TexturePackSampler)
    ax_device_callable explicit TexturePackSampler(const texture_pack &texture_p) : tpack(texture_p) {}

    ax_device_callable ax_no_discard glm::vec4 emissive(float u, float v, const texturing::texture_data_aggregate_s &sample_data) const {
      using namespace math::texture;
      if (tpack.emissive.get()) {
        glm::vec4 value = tpack.emissive.sample(u, v, sample_data);
        return {rgb_uint2float(value.r), rgb_uint2float(value.g), rgb_uint2float(value.b), value.a};
      }
      return glm::vec4(0.f);
    }

    ax_device_callable ax_no_discard glm::vec4 albedo(float u, float v, const texturing::texture_data_aggregate_s &sample_data) const {
      using namespace math::texture;
      if (tpack.albedo.get()) {
        glm::vec4 value = tpack.albedo.sample(u, v, sample_data);
        return {rgb_uint2float(value.r), rgb_uint2float(value.g), rgb_uint2float(value.b), value.a};
      }
      return glm::vec4(1.f);
    }

    ax_device_callable ax_no_discard glm::vec4 metallic(float u, float v, const texturing::texture_data_aggregate_s &sample_data) const {
      using namespace math::texture;
      if (tpack.metallic.get()) {
        glm::vec4 value = tpack.metallic.sample(u, v, sample_data);
        return {rgb_uint2float(value.r), rgb_uint2float(value.g), rgb_uint2float(value.b), value.a};
      }
      return glm::vec4(0.f);
    }

    ax_device_callable ax_no_discard glm::vec4 roughness(float u, float v, const texturing::texture_data_aggregate_s &sample_data) const {
      using namespace math::texture;
      if (tpack.roughness.get()) {
        glm::vec4 value = tpack.roughness.sample(u, v, sample_data);
        return {rgb_uint2float(value.r), rgb_uint2float(value.g), rgb_uint2float(value.b), value.a};
      }
      return glm::vec4(0.f);
    }

    ax_device_callable ax_no_discard glm::vec4 normal(float u, float v, const texturing::texture_data_aggregate_s &sample_data) const {
      using namespace math::texture;
      if (tpack.normalmap.get()) {
        glm::vec4 value = tpack.normalmap.sample(u, v, sample_data);
        return {rgb_uint2float(value.r), rgb_uint2float(value.g), rgb_uint2float(value.b), value.a};
      }
      return glm::vec4(0.f, 0.f, 1.f, 0.f);
    }

    ax_device_callable ax_no_discard glm::vec4 ao(float u, float v, const texturing::texture_data_aggregate_s &sample_data) const {
      using namespace math::texture;
      if (tpack.ao.get()) {
        glm::vec4 value = tpack.ao.sample(u, v, sample_data);
        return {rgb_uint2float(value.r), rgb_uint2float(value.g), rgb_uint2float(value.b), value.a};
      }
      return glm::vec4(0.f);
    }

    ax_device_callable ax_no_discard glm::vec4 opacity(float u, float v, const texturing::texture_data_aggregate_s &sample_data) const {
      using namespace math::texture;
      if (tpack.opacity.get()) {
        glm::vec4 value = tpack.opacity.sample(u, v, sample_data);
        return {rgb_uint2float(value.r), rgb_uint2float(value.g), rgb_uint2float(value.b), value.a};
      }
      return glm::vec4(0.f);
    }

    ax_device_callable ax_no_discard glm::vec4 specular(float u, float v, const texturing::texture_data_aggregate_s &sample_data) const {
      using namespace math::texture;
      if (tpack.specular.get()) {
        glm::vec4 value = tpack.specular.sample(u, v, sample_data);
        return {rgb_uint2float(value.r), rgb_uint2float(value.g), rgb_uint2float(value.b), value.a};
      }
      return glm::vec4(0.f);
    }
  };

}  // namespace nova::material

#endif
