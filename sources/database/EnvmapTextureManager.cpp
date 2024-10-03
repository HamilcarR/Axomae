#include "EnvmapTextureManager.h"
#include "Config.h"
#include "Loader.h"
#include "RenderPipeline.h"
#include "Scene.h"
#include "internal/common/exception/GenericException.h"
#include "internal/debug/Logger.h"
#include "internal/macro/project_macros.h"

std::vector<texture::envmap::EnvmapTextureGroup> EnvmapTextureManager::bakes_id{};

// TODO : read config values from file/cmd
static texture::envmap::EnvmapBakingConfig generate_config() {
  texture::envmap::EnvmapBakingConfig config{};
  config.skybox_dim.width = 2048;
  config.skybox_dim.height = 2048;
  config.irradiance_dim.width = 64;
  config.irradiance_dim.height = 64;
  config.prefilter_dim.width = 512;
  config.prefilter_dim.height = 512;
  config.base_env_dim_upscale = 4096;
  config.prefilter_mip_maps = 10;
  config.base_samples = 500;
  config.lut.width = 256;
  config.lut.height = 256;
  config.sampling_factor_per_mips = 2;
  return config;
}

EnvmapTextureManager::EnvmapTextureManager(ResourceDatabaseManager &resource_database_,
                                           const Dim2 &screen_dimensions,
                                           unsigned int &default_id,
                                           RenderPipeline &rdp,
                                           Scene *scene_,
                                           int policy)
    : resource_database(&resource_database_),
      screen_dim(screen_dimensions),
      cuda_process(false),
      default_framebuffer_id(&default_id),
      render_pipeline(&rdp),
      scene(scene_),
      update_policy_flag(policy) {
  resource_database->getHdrDatabase()->attach(*this);
  config = generate_config();
  if (scene)
    skybox_mesh = &scene->getSkybox();
}

static void createFurnace(HdrImageDatabase &database) {
  image::Metadata metadata;
  metadata.name = "Furnace.hdr";
  metadata.width = 256;
  metadata.height = 256;
  metadata.channels = 4;
  metadata.color_corrected = true;
  metadata.is_hdr = true;
  std::vector<float> image_data(metadata.width * metadata.height * metadata.channels, 0.1f);
  database::image::store<float>(database, true, image_data, metadata);
}

static void createBlack(HdrImageDatabase &database) {
  image::Metadata metadata;
  metadata.name = "Black.hdr";
  metadata.width = 256;
  metadata.height = 256;
  metadata.channels = 3;
  metadata.color_corrected = true;
  metadata.is_hdr = true;
  std::vector<float> image_data(metadata.width * metadata.height * metadata.channels, 0.f);
  database::image::store<float>(database, true, image_data, metadata);
}

void EnvmapTextureManager::initializeDefaultEnvmap(ApplicationConfig *conf) {
  /* Need to compute the LUT beforehand , because of the add event in the HDR database*/
  if (resource_database->getTextureDatabase()->getTexturesByType(GenericTexture::BRDFLUT).empty())
    resource_database->getTextureDatabase()->setPersistence(
        render_pipeline->generateBRDFLookupTexture(config.lut.width, config.lut.height, screen_dim));
  /*
   * Adding the default envmap :
   * This needs to always have a default state
   */
  if (resource_database->getHdrDatabase()->empty()) {
    try {
      IO::Loader loader(nullptr);
      loader.loadHdrEnvmap(config.default_envmap_path.c_str());
    } catch (const exception::GenericException &e) {
      LOG(e.what(), LogLevel::ERROR);
      createBlack(*resource_database->getHdrDatabase());
      createFurnace(*resource_database->getHdrDatabase());
    }
  }
  current = bakes_id.back();
}

void EnvmapTextureManager::notified(observer::Data<Message *> &message) {
  switch (message.data->getOperation()) {
    case Message::ADD:
      if (update_policy_flag & ADD) {
        addToCollection(message.data->getIndex());
      }
      updateCurrent(message.data->getIndex());
      break;
    case Message::DELETE:
      if (update_policy_flag & DELETE) {
        deleteFromCollection(message.data->getIndex());
      }
      updateCurrent(message.data->getIndex());
      break;
    case Message::SELECTED:
      if (update_policy_flag & SELECTED) {
        updateCurrent(message.data->getIndex());
      }
      break;
    default:
      break;
  }
}

void EnvmapTextureManager::updateCurrent(int index) {
  AX_ASSERT(index < bakes_id.size(), "Environment map index selected discrepancy with the Environment map manager id collection. ");
  current = bakes_id[index];
  if (scene)
    scene->switchEnvmap(current.cubemap_id, current.irradiance_id, current.prefiltered_id, current.lut_id);
}

void EnvmapTextureManager::next() {}

void EnvmapTextureManager::previous() {}

void EnvmapTextureManager::deleteFromCollection(int index) {
  for (auto it = bakes_id.begin(); it != bakes_id.end(); it++)
    if (it->equirect_id == index)
      bakes_id.erase(it);
}

static TextureData texture_metadata(image::ThumbnailImageHolder<float> *raw_image_data) {
  TextureData envmap;
  envmap.width = raw_image_data->metadata.width;
  envmap.height = raw_image_data->metadata.height;
  envmap.name = raw_image_data->metadata.name;
  envmap.data_type = GenericTexture::FLOAT;
  envmap.internal_format = GenericTexture::RGBA32F;
  envmap.data_format = GenericTexture::RGBA;
  envmap.nb_components = raw_image_data->metadata.channels;
  envmap.f_data = raw_image_data->data;
  return envmap;
}

void EnvmapTextureManager::addToCollection(int index) {
  TextureDatabase *texture_database = resource_database->getTextureDatabase();
  HdrImageDatabase *image_database = resource_database->getHdrDatabase();
  texture::envmap::EnvmapTextureGroup texgroup{};
  image::ThumbnailImageHolder<float> *raw_image_data = image_database->get(index);
  AX_ASSERT(raw_image_data, "Index returns invalid pointer.");
  texgroup.equirect_id = index;
  texgroup.metadata = raw_image_data;
  TextureData envmap = texture_metadata(raw_image_data);
  auto result = database::texture::store<EnvironmentMap2DTexture>(*texture_database, false, &envmap);
  AX_ASSERT_NOTNULL(result.object);
  if (!cuda_process) {
    texgroup.cubemap_id = render_pipeline->bakeEnvmapToCubemap(
        result.object, *skybox_mesh, config.skybox_dim.width, config.skybox_dim.height, screen_dim);
    texgroup.irradiance_id = render_pipeline->bakeIrradianceCubemap(
        texgroup.cubemap_id, config.irradiance_dim.width, config.irradiance_dim.height, screen_dim);
    texgroup.prefiltered_id = render_pipeline->preFilterEnvmap(texgroup.cubemap_id,
                                                               config.base_env_dim_upscale,
                                                               config.prefilter_dim.width,
                                                               config.prefilter_dim.height,
                                                               config.prefilter_mip_maps,
                                                               config.base_samples,
                                                               config.sampling_factor_per_mips,
                                                               screen_dim);
    texture_database->setPersistence(texgroup.cubemap_id);
    texture_database->setPersistence(texgroup.irradiance_id);
    texture_database->setPersistence(texgroup.prefiltered_id);
  }
  texgroup.lut_id = texture_database->getTexturesByType(GenericTexture::BRDFLUT)[0].id;
  bakes_id.push_back(texgroup);
}
