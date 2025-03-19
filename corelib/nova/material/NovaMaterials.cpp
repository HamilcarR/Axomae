#include "NovaMaterials.h"
#include "ray/Hitable.h"
#include "ray/Ray.h"
#include "sampler/Sampler.h"
#include "texturing/nova_texturing.h"
#include <internal/common/math/utils_3D.h>
#include <internal/macro/project_macros.h>
namespace nova::material {

  class TexturePackSampler {
   private:
    texture_pack tpack{};

   public:
    CLASS_DCM(TexturePackSampler)
    ax_device_callable explicit TexturePackSampler(const texture_pack &texture_p) : tpack(texture_p) {}

    ax_device_callable ax_no_discard glm::vec4 emissive(float u, float v, const texturing::texture_data_aggregate_s &sample_data) const {
      using namespace math::texture;
      if (tpack.emissive) {
        glm::vec4 value = tpack.emissive->sample(u, v, sample_data);
        return {rgb_uint2float(value.r), rgb_uint2float(value.g), rgb_uint2float(value.b), value.a};
      }
      return glm::vec4(0.f);
    }

    ax_device_callable ax_no_discard glm::vec4 albedo(float u, float v, const texturing::texture_data_aggregate_s &sample_data) const {
      using namespace math::texture;
      if (tpack.albedo) {
        glm::vec4 value = tpack.albedo->sample(u, v, sample_data);
        return {rgb_uint2float(value.r), rgb_uint2float(value.g), rgb_uint2float(value.b), value.a};
      }
      return glm::vec4(1.f);
    }

    ax_device_callable ax_no_discard glm::vec4 metallic(float u, float v, const texturing::texture_data_aggregate_s &sample_data) const {
      using namespace math::texture;
      if (tpack.metallic) {
        glm::vec4 value = tpack.metallic->sample(u, v, sample_data);
        return {rgb_uint2float(value.r), rgb_uint2float(value.g), rgb_uint2float(value.b), value.a};
      }
      return glm::vec4(0.f);
    }

    ax_device_callable ax_no_discard glm::vec4 roughness(float u, float v, const texturing::texture_data_aggregate_s &sample_data) const {
      using namespace math::texture;
      if (tpack.roughness) {
        glm::vec4 value = tpack.roughness->sample(u, v, sample_data);
        return {rgb_uint2float(value.r), rgb_uint2float(value.g), rgb_uint2float(value.b), value.a};
      }
      return glm::vec4(0.f);
    }

    ax_device_callable ax_no_discard glm::vec4 normal(float u, float v, const texturing::texture_data_aggregate_s &sample_data) const {
      using namespace math::texture;
      if (tpack.normalmap) {
        glm::vec4 value = tpack.normalmap->sample(u, v, sample_data);
        return {rgb_uint2float(value.r), rgb_uint2float(value.g), rgb_uint2float(value.b), value.a};
      }
      return glm::vec4(0.f, 0.f, 1.f, 0.f);
    }

    ax_device_callable ax_no_discard glm::vec4 ao(float u, float v, const texturing::texture_data_aggregate_s &sample_data) const {
      using namespace math::texture;
      if (tpack.ao) {
        glm::vec4 value = tpack.ao->sample(u, v, sample_data);
        return {rgb_uint2float(value.r), rgb_uint2float(value.g), rgb_uint2float(value.b), value.a};
      }
      return glm::vec4(0.f);
    }

    ax_device_callable ax_no_discard glm::vec4 opacity(float u, float v, const texturing::texture_data_aggregate_s &sample_data) const {
      using namespace math::texture;
      if (tpack.opacity) {
        glm::vec4 value = tpack.opacity->sample(u, v, sample_data);
        return {rgb_uint2float(value.r), rgb_uint2float(value.g), rgb_uint2float(value.b), value.a};
      }
      return glm::vec4(0.f);
    }

    ax_device_callable ax_no_discard glm::vec4 specular(float u, float v, const texturing::texture_data_aggregate_s &sample_data) const {
      using namespace math::texture;
      if (tpack.specular) {
        glm::vec4 value = tpack.specular->sample(u, v, sample_data);
        return {rgb_uint2float(value.r), rgb_uint2float(value.g), rgb_uint2float(value.b), value.a};
      }
      return glm::vec4(0.f);
    }
  };

  constexpr float INTERSECT_OFFSET = 1e-4f;

  ax_device_callable glm::vec3 compute_map_normal(const hit_data &hit_d,
                                                  const TexturePackSampler &t_pack,
                                                  const glm::mat3 &tbn,
                                                  texturing::texture_data_aggregate_s &sample_data) {
    const glm::vec3 map_normal = t_pack.normal(hit_d.u, hit_d.v, sample_data) * 2.f - 1.f;
    return glm::normalize(tbn * map_normal);
  }

  NovaDiffuseMaterial::NovaDiffuseMaterial(const texture_pack &texture) { t_pack = texture; }

  ax_device_callable glm::vec3 hemi_sample(const glm::mat3 & /*tbn*/, sampler::SamplerInterface &sampler) {
    using namespace math::random;
    using namespace math::spherical;
    auto random_sample = sampler.sample();
    return glm::normalize(random_sample);
  }

  bool NovaDiffuseMaterial::scatter(
      const Ray & /*in*/, Ray &out, hit_data &hit_d, sampler::SamplerInterface &sampler, shading_data_s &mat_ctx) const {
    TexturePackSampler texture_pack_sampler(t_pack);
    const glm::mat3 tbn = math::geometry::construct_tbn(hit_d.normal, hit_d.tangent, hit_d.bitangent);
    glm::vec3 normal = compute_map_normal(hit_d, texture_pack_sampler, tbn, *mat_ctx.texture_aggregate);
    out.direction = hemi_sample(tbn, sampler);
    out.origin = hit_d.position + hit_d.normal * INTERSECT_OFFSET;
    AX_ASSERT_NOTNULL(mat_ctx.texture_aggregate);
    mat_ctx.texture_aggregate->geometric_data.sampling_vector = hit_d.position;
    hit_d.attenuation = texture_pack_sampler.albedo(hit_d.u, hit_d.v, *mat_ctx.texture_aggregate);
    hit_d.normal = normal;
    hit_d.emissive = texture_pack_sampler.emissive(hit_d.u, hit_d.v, *mat_ctx.texture_aggregate);
    return glm::dot(normal, out.direction) > 0;
  }

  bool NovaMaterialInterface::scatter(const Ray &in, Ray &out, hit_data &hit_d, sampler::SamplerInterface &sampler, shading_data_s &mat_ctx) const {
    auto disp = [&](auto material) { return material->scatter(in, out, hit_d, sampler, mat_ctx); };
    return dispatch(disp);
  }

  /********************************************************************************************************************************/

  NovaConductorMaterial::NovaConductorMaterial(const texture_pack &texture) { t_pack = texture; }
  NovaConductorMaterial::NovaConductorMaterial(const texture_pack &texture, float fuzz_) {
    fuzz = fuzz_;
    t_pack = texture;
  }

  bool NovaConductorMaterial::scatter(const Ray &in, Ray &out, hit_data &hit_d, sampler::SamplerInterface &sampler, shading_data_s &mat_ctx) const {
    TexturePackSampler texture_pack_sampler(t_pack);
    const glm::mat3 tbn = math::geometry::construct_tbn(hit_d.normal, hit_d.tangent, hit_d.bitangent);
    glm::vec3 normal = compute_map_normal(hit_d, texture_pack_sampler, tbn, *mat_ctx.texture_aggregate);
    glm::vec3 reflected = glm::reflect(in.direction, normal);
    out.origin = hit_d.position + hit_d.normal * INTERSECT_OFFSET;
    out.direction = glm::normalize(reflected + hemi_sample(tbn, sampler) * fuzz);
    AX_ASSERT(!ISNAN(out.direction), "");
    AX_ASSERT_NOTNULL(mat_ctx.texture_aggregate);
    mat_ctx.texture_aggregate->geometric_data.sampling_vector = hit_d.position;
    hit_d.attenuation = texture_pack_sampler.albedo(hit_d.u, hit_d.v, *mat_ctx.texture_aggregate);
    hit_d.emissive = texture_pack_sampler.emissive(hit_d.u, hit_d.v, *mat_ctx.texture_aggregate);
    hit_d.normal = normal;
    return glm::dot(out.direction, normal) > 0.f;
  }

  /********************************************************************************************************************************/

  NovaDielectricMaterial::NovaDielectricMaterial(const texture_pack &texture) {
    t_pack = texture;
    eta = 1.f;
  }
  NovaDielectricMaterial::NovaDielectricMaterial(const texture_pack &texture, float ior) {
    t_pack = texture;
    eta = ior;
  }
  ax_device_callable_inlined bool refract(const glm::vec3 &v, const glm::vec3 &n, float eta, glm::vec3 &refracted) {
    glm::vec3 inc = glm::normalize(v);
    float dt = glm::dot(inc, n);
    float discriminant = 1.f - eta * eta * (1 - dt * dt);
    if (discriminant > 1e-4f) {
      refracted = glm::refract(v, n, eta);
      return true;
    }
    return false;
  }

  ax_device_callable_inlined float schlick(float cosine, float eta) {
    float r0 = (1 - eta) / (1 + eta);
    r0 *= r0;
    return r0 + (1 - r0) * std::pow((1 - cosine), 5);
  }

  bool NovaDielectricMaterial::scatter(const Ray &in, Ray &out, hit_data &hit_d, sampler::SamplerInterface &sampler, shading_data_s &mat_ctx) const {
    TexturePackSampler texture_pack_sampler(t_pack);
    const glm::mat3 tbn = math::geometry::construct_tbn(hit_d.normal, hit_d.tangent, hit_d.bitangent);
    glm::vec3 perturbed_normal = compute_map_normal(hit_d, texture_pack_sampler, tbn, *mat_ctx.texture_aggregate);
    glm::vec3 original_perturbed_normal = perturbed_normal;
    glm::vec3 direction = glm::normalize(in.direction);
    const glm::vec3 reflected = glm::reflect(direction, perturbed_normal);
    out.origin = hit_d.position;
    AX_ASSERT_NOTNULL(mat_ctx.texture_aggregate);
    mat_ctx.texture_aggregate->geometric_data.sampling_vector = out.origin;
    hit_d.attenuation = texture_pack_sampler.albedo(hit_d.u, hit_d.v, *mat_ctx.texture_aggregate);
    hit_d.emissive = texture_pack_sampler.emissive(hit_d.u, hit_d.v, *mat_ctx.texture_aggregate);
    float index = eta;
    float reflect_prob = 0.f;
    float cosine = 0.f;
    if (glm::dot(direction, perturbed_normal) > 0) {
      original_perturbed_normal = -original_perturbed_normal;
      cosine = glm::dot(direction, perturbed_normal) * eta / glm::length(direction);
    } else {
      index = 1.f / eta;
      cosine = -glm::dot(direction, perturbed_normal) / glm::length(direction);
    }
    glm::vec3 refracted;
    if (refract(direction, original_perturbed_normal, index, refracted))
      reflect_prob = schlick(cosine, eta);
    else
      reflect_prob = 1.f;
    float sampler_random = (sampler.sample().x + 1) / 2;
    if (sampler_random < reflect_prob)
      out.direction = glm::normalize(reflected);
    else
      out.direction = glm::normalize(refracted);
    hit_d.normal = perturbed_normal;
    AX_ASSERT(!ISNAN(out.direction), "");
    return true;
  }
}  // namespace nova::material
