#include "material/NovaMaterials.h"
#include "ray/Hitable.h"
#include "ray/Ray.h"
#include "sampler/Sampler.h"
#include "texturing/NovaTextureInterface.h"
#include "texturing/TextureContext.h"
#include "texturing/texture_datastructures.h"
#include <unit_test/Test.h>

constexpr int MAX_ITER = 30;
constexpr uint32_t BLUE = 0x0000FF00;
constexpr uint32_t GREEN = 0x00FF0000;
constexpr uint32_t RED = 0x00FF0000;

class TextureCtxBuilder {
  std::vector<U32Texture> textures;
  nova::texturing::u32tex_shared_views_s u32_views;
  nova::texturing::TextureBundleViews bundle;

 public:
  TextureCtxBuilder() = default;
  void addTexture(const uint32_t *buffer, int w, int h, int chan) {
    U32Texture tex;
    tex.raw_data = axstd::span(buffer, w * h * chan);
    tex.width = w;
    tex.height = h;
    tex.channels = chan;
    tex.is_rgba = false;
    textures.push_back(tex);
  }

  nova::texturing::TextureCtx getTextureContext() {
    u32_views.u32_host = textures;
    bundle = nova::texturing::TextureBundleViews(u32_views);
    nova::texturing::TextureCtx ctx(bundle, false);
    return ctx;
  }
};

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
  return tpack;
}

TEST(NovaDiffuseMaterialTest, scatter_direction) {
  math::random::CPUPseudoRandomGenerator generator;
  const uint32_t buffer[4] = {RED, GREEN, BLUE, BLUE};
  nova::texturing::ImageTexture img(0);
  TextureCtxBuilder ctxBuilder;
  ctxBuilder.addTexture(buffer, 2, 2, 4);
  auto ctx = ctxBuilder.getTextureContext();
  nova::texturing::texture_data_aggregate_s tex_aggregate;
  tex_aggregate.texture_ctx = &ctx;
  nova::material::shading_data_s shading;
  shading.texture_aggregate = &tex_aggregate;

  nova::material::texture_pack tpack = setup_tpack(&img);
  nova::material::NovaDiffuseMaterial diffuse_material(tpack);
  nova::hit_data hit_data{};
  init_tbn(hit_data);

  /* We set up a horizontal plane , with y as normal , centered on 0 */
  hit_data.position = {0.f, 0.f, 0.f};
  const nova::Ray ray(glm::vec3(-1.f, 1.f, 0.f), glm::vec3(1.f, -1.f, 0.f));
  math::random::CPUQuasiRandomGenerator quasi_random(1000, 3);
  nova::sampler::SobolSampler sobol = nova::sampler::SobolSampler(quasi_random);
  nova::sampler::SamplerInterface sampler = &sobol;
  nova::Ray out{};

  for (int i = 0; i < MAX_ITER; i++) {
    hit_data.u = (float)generator.nrandf(0, 1);
    hit_data.v = (float)generator.nrandf(0, 1);
    if (diffuse_material.scatter(ray, out, hit_data, sampler, shading)) {
      const glm::vec3 transformed_normal_computed = hit_data.normal;
      ASSERT_GT(glm::dot(transformed_normal_computed, out.direction), 0);
    }
  }
}

TEST(NovaDiffuseMaterialTest, sample_normal) {
  const uint32_t buffer[4] = {RED, GREEN, BLUE, BLUE};
  nova::texturing::ImageTexture img(0);
  TextureCtxBuilder ctxBuilder;
  ctxBuilder.addTexture(buffer, 2, 2, 4);
  auto ctx = ctxBuilder.getTextureContext();
  nova::texturing::texture_data_aggregate_s tex_aggregate;
  tex_aggregate.texture_ctx = &ctx;
  nova::material::shading_data_s shading;
  shading.texture_aggregate = &tex_aggregate;

  nova::material::texture_pack tpack = setup_tpack(&img);
  nova::material::NovaDiffuseMaterial diffuse_material(tpack);
  nova::hit_data hit_data{};
  init_tbn(hit_data);

  const glm::mat3 tbn(hit_data.tangent, hit_data.bitangent, hit_data.normal);
  hit_data.u = hit_data.v = 0.f;
  /* We set up a horizontal plane , with y as normal , centered on 0 */
  hit_data.position = {0.f, 0.f, 0.f};
  const nova::Ray ray(glm::vec3(-1.f, 1.f, 0.f), glm::vec3(1.f, -1.f, 0.f));
  math::random::CPUQuasiRandomGenerator quasi_random(1000, 3);
  nova::sampler::SobolSampler sobol = nova::sampler::SobolSampler(quasi_random);
  nova::sampler::SamplerInterface sampler = &sobol;
  nova::Ray out{};
  ASSERT_TRUE(diffuse_material.scatter(ray, out, hit_data, sampler, shading));
  const glm::vec3 transformed_normal_computed = hit_data.normal;
  const glm::vec3 tangent_space_normal = glm::vec3(math::texture::rgb_uint2float(RED >> 16), 0, 0) * 2.f - 1.f;
  const glm::vec3 world_space_normal = glm::normalize(tbn * tangent_space_normal);
  ASSERT_EQ(transformed_normal_computed, world_space_normal);
}
