#include "EnvmapTextureManager.h"
#include "Axomae_macros.h"
#include "Loader.h"
#include "Logger.h"
#include "Mesh.h"
#include "RenderPipeline.h"
#include "Scene.h"

/* Use this for now , later we read config values from file*/
static texture::envmap::EnvmapBakingConfig generate_config() {
  texture::envmap::EnvmapBakingConfig config{};
  config.skybox_dim.width = 2048;
  config.skybox_dim.height = 2048;
  config.irradiance_dim.width = 64;
  config.irradiance_dim.height = 64;
  config.prefilter_dim.width = 512;
  config.prefilter_dim.height = 512;
  config.base_env_dim_upscale = 2048;
  config.prefilter_mip_maps = 10;
  config.base_samples = 100;
  config.lut.width = 256;
  config.lut.height = 256;
  config.sampling_factor_per_mips = 2;
  config.default_envmap_path = "test1.hdr";
  return config;
}

EnvmapTextureManager::EnvmapTextureManager(
    ResourceDatabaseManager &resource_database_, Dim2 &screen_dimensions, unsigned int &default_id, RenderPipeline &rdp, Scene &scene_)
    : resource_database(resource_database_),
      screen_dim(screen_dimensions),
      cuda_process(false),
      default_framebuffer_id(default_id),
      render_pipeline(rdp),
      scene(scene_) {
  resource_database.getHdrDatabase().attach(*this);
  config = generate_config();

  skybox_mesh = &scene.getSkybox();
}

void EnvmapTextureManager::initializeLUT() {

  /* Need to compute the LUT beforehand , because of the add event in the HDR database*/
  if (resource_database.getTextureDatabase().getTexturesByType(Texture::BRDFLUT).empty())
    resource_database.getTextureDatabase().setPersistence(render_pipeline.generateBRDFLookupTexture(config.lut.width, config.lut.height, screen_dim));
  /*Adding the default envmap*/
  if (resource_database.getHdrDatabase().empty()) {
    try {
      IO::Loader::loadHdr(config.default_envmap_path.c_str());
      current = bakes_id.back();
    } catch (GenericException &e) {
      LOG(e.what(), LogLevel::ERROR);
    }
  }
}

void EnvmapTextureManager::notified(observer::Data<EnvmapTextureManager::Message *> &message) {
  switch (message.data->getOperation()) {
    case Message::ADD:
      addToCollection(message.data->getIndex());
      break;
    case Message::DELETE:
      deleteFromCollection(message.data->getIndex());
      break;
    case Message::SELECTED:
      updateCurrent(message.data->getIndex());
    default:
      return;
  }
}

void EnvmapTextureManager::updateCurrent(int index) {
  assert(static_cast<unsigned>(index) < bakes_id.size());
  current = bakes_id[index];
  scene.switchEnvmap(current.cubemap_id, current.irradiance_id, current.prefiltered_id, current.lut_id);
}

void EnvmapTextureManager::next() {}

void EnvmapTextureManager::previous() {}

void EnvmapTextureManager::deleteFromCollection(int index) {
  for (auto it = bakes_id.begin(); it != bakes_id.end(); it++)
    if (it->equirect_id == index)
      bakes_id.erase(it);
}

static TextureData texture_metadata(image::RawImageHolder<float> *raw_image_data) {
  TextureData envmap;
  envmap.width = raw_image_data->metadata().width;
  envmap.height = raw_image_data->metadata().height;
  envmap.name = raw_image_data->metadata().name;
  envmap.data_type = Texture::FLOAT;
  envmap.internal_format = Texture::RGB32F;
  envmap.data_format = Texture::RGB;
  envmap.nb_components = raw_image_data->metadata().channels;
  envmap.f_data = raw_image_data->data;
  return envmap;
}

void EnvmapTextureManager::addToCollection(int index) {
  auto &texture_database = resource_database.getTextureDatabase();
  auto &image_database = resource_database.getHdrDatabase();
  texture::envmap::EnvmapTextureGroup texgroup{};
  image::RawImageHolder<float> *raw_image_data = image_database.get(index);
  assert(raw_image_data);
  texgroup.equirect_id = index;
  TextureData envmap = texture_metadata(raw_image_data);
  auto result = database::texture::store<EnvironmentMap2DTexture>(texture_database, false, &envmap);
  assert(result.object);
  if (!cuda_process) {
    texgroup.cubemap_id = render_pipeline.bakeEnvmapToCubemap(
        result.object, *skybox_mesh, config.skybox_dim.width, config.skybox_dim.height, screen_dim);
    texgroup.irradiance_id = render_pipeline.bakeIrradianceCubemap(
        texgroup.cubemap_id, config.irradiance_dim.width, config.irradiance_dim.height, screen_dim);
    texgroup.prefiltered_id = render_pipeline.preFilterEnvmap(texgroup.cubemap_id,
                                                              config.base_env_dim_upscale,
                                                              config.prefilter_dim.width,
                                                              config.prefilter_dim.height,
                                                              config.prefilter_mip_maps,
                                                              config.base_samples,
                                                              config.sampling_factor_per_mips,
                                                              screen_dim);
    texture_database.setPersistence(texgroup.cubemap_id);
    texture_database.setPersistence(texgroup.irradiance_id);
    texture_database.setPersistence(texgroup.prefiltered_id);
  } else {
    /* Cuda baking here*/
  }
  texgroup.lut_id = texture_database.getTexturesByType(Texture::BRDFLUT)[0].id;
  bakes_id.push_back(texgroup);
}
