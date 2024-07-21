#include "Test.h"

#include <material/NovaMaterials.h>
#include <ray/Ray.h>
#include <texturing/nova_texturing.h>

constexpr int MAX_ITER = 30;
constexpr uint32_t BLUE = 0x0000FF00;
constexpr uint32_t GREEN = 0x00FF0000;
constexpr uint32_t RED = 0x00FF0000;

static nova::texturing::ImageTexture generate_image_texture(const uint32_t *buffer, int w, int h, int chan, bool rgba = false) {
  return {buffer, w, h, chan, rgba};
}
static void init_tbn(nova::hit_data &hit_data) {
  hit_data.tangent = {1.f, 0.f, 0.f};
  hit_data.normal = {0.f, 1.f, 0.f};
  hit_data.bitangent = {0.f, 0.f, 1.f};
}

static nova::material::texture_pack setup_tpack(const nova::texturing::ImageTexture *img) {
  nova::material::texture_pack tpack{};
  tpack.albedo = img;
  tpack.normalmap = img;
  tpack.ao = img;
  tpack.emissive = img;
  tpack.metallic = img;
  tpack.roughness = img;
}

TEST(NovaDiffuseMaterialTest, sample_normal) {
  const uint32_t buffer[4] = {RED, GREEN, BLUE, BLUE};
  nova::texturing::ImageTexture img = generate_image_texture((const uint32_t *)buffer, 2, 2, 4);
  nova::material::texture_pack tpack = setup_tpack(&img);
  nova::material::NovaDiffuseMaterial diffuse_material(tpack);
  nova::hit_data hit_data{};
  init_tbn(hit_data);

  const glm::mat3 tbn(hit_data.tangent, hit_data.bitangent, hit_data.normal);
  hit_data.u = hit_data.v = 0.f;
  /* We set up a horizontal plane , with y as normal , centered on 0 */
  hit_data.position = {0.f, 0.f, 0.f};
  const nova::Ray ray(glm::vec3(-1.f, 1.f, 0.f), glm::vec3(1.f, -1.f, 0.f));
  nova::sampler::SamplerInterface sampler = nova::sampler::SobolSampler(1000, 3);
  nova::Ray out{};
  ASSERT_TRUE(diffuse_material.scatter(ray, out, hit_data, sampler));
  const glm::vec3 transformed_normal_computed = hit_data.normal;
  const glm::vec3 tangent_space_normal = glm::vec3(math::texture::rgb_uint2float(RED >> 16), 0, 0) * 2.f - 1.f;
  const glm::vec3 world_space_normal = glm::normalize(tbn * tangent_space_normal);
  ASSERT_EQ(transformed_normal_computed, world_space_normal);
}