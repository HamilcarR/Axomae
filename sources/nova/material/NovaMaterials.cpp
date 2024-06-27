#include "NovaMaterials.h"
#include "ray/Ray.h"

namespace nova::material {

  class TexturePackSampler {
   private:
    texture_pack tpack;

   public:
    CLASS_CM(TexturePackSampler)
    explicit TexturePackSampler(const texture_pack &texture_p) : tpack(texture_p) {}

    [[nodiscard]] glm::vec4 emissive(float u, float v, const texturing::texture_sample_data &sample_data) const {
      using namespace math::texture;
      if (tpack.emissive) {
        glm::vec4 value = tpack.emissive->sample(u, v, sample_data);
        return {rgb_uint2float(value.r), rgb_uint2float(value.g), rgb_uint2float(value.b), value.a};
      }
      return glm::vec4(0.f);
    }

    [[nodiscard]] glm::vec4 albedo(float u, float v, const texturing::texture_sample_data &sample_data) const {
      using namespace math::texture;
      if (tpack.albedo) {
        glm::vec4 value = tpack.albedo->sample(u, v, sample_data);
        return {rgb_uint2float(value.r), rgb_uint2float(value.g), rgb_uint2float(value.b), value.a};
      }
      return glm::vec4(0.f);
    }

    [[nodiscard]] glm::vec4 metallic(float u, float v, const texturing::texture_sample_data &sample_data) const {
      using namespace math::texture;
      if (tpack.metallic) {
        glm::vec4 value = tpack.metallic->sample(u, v, sample_data);
        return {rgb_uint2float(value.r), rgb_uint2float(value.g), rgb_uint2float(value.b), value.a};
      }
      return glm::vec4(0.f);
    }

    [[nodiscard]] glm::vec4 roughness(float u, float v, const texturing::texture_sample_data &sample_data) const {
      using namespace math::texture;
      if (tpack.roughness) {
        glm::vec4 value = tpack.roughness->sample(u, v, sample_data);
        return {rgb_uint2float(value.r), rgb_uint2float(value.g), rgb_uint2float(value.b), value.a};
      }
      return glm::vec4(0.f);
    }

    [[nodiscard]] glm::vec4 normal(float u, float v, const texturing::texture_sample_data &sample_data) const {
      using namespace math::texture;
      if (tpack.normalmap) {
        glm::vec4 value = tpack.normalmap->sample(u, v, sample_data);
        return {rgb_uint2float(value.r), rgb_uint2float(value.g), rgb_uint2float(value.b), value.a};
      }
      return glm::vec4(0.f, 0.f, 1.f, 0.f);
    }

    [[nodiscard]] glm::vec4 ao(float u, float v, const texturing::texture_sample_data &sample_data) const {
      using namespace math::texture;
      if (tpack.ao) {
        glm::vec4 value = tpack.ao->sample(u, v, sample_data);
        return {rgb_uint2float(value.r), rgb_uint2float(value.g), rgb_uint2float(value.b), value.a};
      }
      return glm::vec4(0.f);
    }
  };

  static glm::mat3 construct_tbn(const glm::vec3 &normal, const glm::vec3 &tangent, const glm::vec3 &bitangent) {
    return {tangent, bitangent, normal};
  }

  static glm::vec3 compute_map_normal(const hit_data &hit_d, const TexturePackSampler &sampler) {
    const glm::mat3 tbn = construct_tbn(hit_d.normal, hit_d.tangent, hit_d.bitangent);
    const glm::vec3 map_normal = sampler.normal(hit_d.u, hit_d.v, {}) * 2.f - 1.f;
    return glm::normalize(tbn * map_normal);
  }

  NovaDiffuseMaterial::NovaDiffuseMaterial(const texture_pack &texture) { t_pack = texture; }

  bool NovaDiffuseMaterial::scatter(const Ray &in, Ray &out, hit_data &hit_d) const {
    TexturePackSampler sampler(t_pack);
    glm::vec3 normal = compute_map_normal(hit_d, sampler);
    glm::vec3 hemi_rand = math::spherical::rand_p_sphere();
    glm::vec3 scattered = hit_d.position + normal + hemi_rand;
    out.origin = hit_d.position;
    out.direction = glm::normalize(scattered);
    texturing::texture_sample_data sample_data{hit_d.position};
    hit_d.attenuation = sampler.albedo(hit_d.u, hit_d.v, sample_data);
    hit_d.emissive = sampler.emissive(hit_d.u, hit_d.v, sample_data);
    hit_d.normal = normal;
    return true;
  }

  /********************************************************************************************************************************/

  NovaConductorMaterial::NovaConductorMaterial(const texture_pack &texture) { t_pack = texture; }
  NovaConductorMaterial::NovaConductorMaterial(const texture_pack &texture, float fuzz_) {
    fuzz = fuzz_;
    t_pack = texture;
  }

  bool NovaConductorMaterial::scatter(const Ray &in, Ray &out, hit_data &hit_d) const {
    TexturePackSampler sampler(t_pack);
    glm::vec3 normal = compute_map_normal(hit_d, sampler);
    glm::vec3 reflected = glm::reflect(in.direction, normal);
    out.origin = hit_d.position;
    out.direction = glm::normalize(reflected + math::spherical::rand_p_sphere() * fuzz);
    texturing::texture_sample_data sample_data{hit_d.position};
    hit_d.attenuation = sampler.albedo(hit_d.u, hit_d.v, {sample_data});
    hit_d.emissive = sampler.emissive(hit_d.u, hit_d.v, sample_data);
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
  static bool refract(const glm::vec3 &v, const glm::vec3 &n, float eta, glm::vec3 &refracted) {
    glm::vec3 inc = glm::normalize(v);
    float dt = glm::dot(inc, n);
    float discriminant = 1.f - eta * eta * (1 - dt * dt);
    if (discriminant > 0) {
      refracted = glm::refract(v, n, eta);
      return true;
    }
    return false;
  }

  static float schlick(float cosine, float eta) {
    float r0 = (1 - eta) / (1 + eta);
    r0 *= r0;
    return r0 + (1 - r0) * std::pow((1 - cosine), 5);
  }

  bool NovaDielectricMaterial::scatter(const Ray &in, Ray &out, hit_data &hit_d) const {
    TexturePackSampler sampler(t_pack);
    hit_d.normal = compute_map_normal(hit_d, sampler);
    glm::vec3 normal = hit_d.normal;
    glm::vec3 direction = glm::normalize(in.direction);
    const glm::vec3 reflected = glm::reflect(direction, hit_d.normal);
    out.origin = hit_d.position;
    texturing::texture_sample_data sample_data{hit_d.position};
    hit_d.attenuation = sampler.albedo(hit_d.u, hit_d.v, sample_data);
    hit_d.emissive = sampler.emissive(hit_d.u, hit_d.v, sample_data);
    float index = eta;
    float reflect_prob = 0.f;
    float cosine = 0.f;
    if (glm::dot(direction, hit_d.normal) > 0) {
      normal = -normal;
      cosine = glm::dot(direction, hit_d.normal) * eta / glm::length(direction);
    } else {
      index = 1.f / eta;
      cosine = -glm::dot(direction, hit_d.normal) / glm::length(direction);
    }
    glm::vec3 refracted;
    if (refract(direction, normal, index, refracted))
      reflect_prob = schlick(cosine, eta);
    else
      reflect_prob = 1.f;

    if (math::random::nrandf(0, 1) < reflect_prob)
      out.direction = glm::normalize(reflected);
    else
      out.direction = glm::normalize(refracted);
    return true;
  }
}  // namespace nova::material