#include "nova_material.h"
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

bool NovaDiffuseMaterial::scatter(const Ray &in, Ray &out, hit_data &hit_d) const {
  glm::vec3 hemi_rand = rand_p_sphere();
  glm::vec3 scattered = hit_d.position + hit_d.normal + hemi_rand;
  out.origin = hit_d.position;
  out.direction = glm::normalize(scattered);
  hit_d.attenuation = albedo;
  return true;
}

bool NovaConductorMaterial::scatter(const Ray &in, Ray &out, hit_data &hit_d) const {
  glm::vec3 reflected = glm::reflect(in.direction, hit_d.normal);
  out.origin = hit_d.position;
  out.direction = glm::normalize(reflected + rand_p_sphere() * fuzz);
  hit_d.attenuation = albedo;
  return glm::dot(out.direction, hit_d.normal) > 0.f;
}