#include "nova_material.h"
#include "texturing/NovaTextures.h"
#include <math.h>
using namespace nova::material;

inline glm::vec3 rand_p_hemisphere() {
  float phi = math::random::nrandf(0.f, 2 * PI);
  float theta = math::random::nrandf(0.f, PI * 0.5f);
  return glm::normalize(math::spherical::sphericalToCartesian(phi, theta));
}

inline glm::vec3 rand_p_sphere() {
  glm::vec3 p;
  do {
    double x = math::random::nrandf(0, 1);
    double y = math::random::nrandf(0, 1);
    double z = math::random::nrandf(0, 1);
    p = 2.f * glm::vec3(x, y, z) - glm::vec3(1, 1, 1);
  } while ((p.x * p.x + p.y * p.y + p.z * p.z) >= 1.f);
  return p;
}

NovaDiffuseMaterial::NovaDiffuseMaterial(texturing::NovaTextureInterface *texture) { albedo = texture; }

bool NovaDiffuseMaterial::scatter(const Ray &in, Ray &out, hit_data &hit_d) const {
  glm::vec3 hemi_rand = rand_p_sphere();
  glm::vec3 scattered = hit_d.position + hit_d.normal + hemi_rand;
  out.origin = hit_d.position;
  out.direction = glm::normalize(scattered);
  texturing::texture_sample_data sample_data{hit_d.position};
  hit_d.attenuation = albedo->sample(hit_d.u, hit_d.v, sample_data);
  return true;
}

/********************************************************************************************************************************/

NovaConductorMaterial::NovaConductorMaterial(texturing::NovaTextureInterface *texture) { albedo = texture; }
NovaConductorMaterial::NovaConductorMaterial(texturing::NovaTextureInterface *texture, float fuzz_) {
  fuzz = fuzz_;
  albedo = texture;
}

bool NovaConductorMaterial::scatter(const Ray &in, Ray &out, hit_data &hit_d) const {
  glm::vec3 reflected = glm::reflect(in.direction, hit_d.normal);
  out.origin = hit_d.position;
  out.direction = glm::normalize(reflected + rand_p_sphere() * fuzz);

  texturing::texture_sample_data sample_data{hit_d.position};
  hit_d.attenuation = albedo->sample(hit_d.u, hit_d.v, {sample_data});
  return glm::dot(out.direction, hit_d.normal) > 0.f;
}

/********************************************************************************************************************************/

NovaDielectricMaterial::NovaDielectricMaterial(texturing::NovaTextureInterface *texture) {
  albedo = texture;
  eta = 1.f;
}
NovaDielectricMaterial::NovaDielectricMaterial(texturing::NovaTextureInterface *texture, float ior) {
  albedo = texture;
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
  glm::vec3 direction = glm::normalize(in.direction);
  const glm::vec3 reflected = glm::reflect(direction, hit_d.normal);
  out.origin = hit_d.position;
  texturing::texture_sample_data sample_data{hit_d.position};
  hit_d.attenuation = albedo->sample(hit_d.u, hit_d.v, sample_data);
  glm::vec3 normal = hit_d.normal;
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
