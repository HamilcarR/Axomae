#ifndef ENVMAPTEXTUREMANAGER_H
#define ENVMAPTEXTUREMANAGER_H
#include "ImageDatabase.h"
#include "Observer.h"
#include "RenderPipeline.h"
#include "ResourceDatabaseManager.h"
#include "TextureDatabase.h"
#include "constants.h"
#include <vector>
/**
 * @file EnvmapTextureManager.h
 *
 */

namespace texture::envmap {
  /*This structure packs all 3 cubemaps IDs used in IBL*/
  struct EnvmapTextureGroup {
    /*TextureDatabase IDs*/
    int cubemap_id;
    int prefiltered_id;
    int irradiance_id;
    /*EnvironmentMap2DTexture ID*/
    int equirect_id;
    int lut_id;
  };

  struct EnvmapBakingConfig {
    Dim2 skybox_dim;
    Dim2 irradiance_dim;
    Dim2 lut;
    Dim2 prefilter_dim;
    int prefilter_mip_maps;
    int sampling_factor_per_mips;
    int base_samples;
    int base_env_dim_upscale;
  };
}  // namespace texture::envmap

/**
 * @class EnvmapTextureManager
 * @brief Tracks the current scene's envmap IDs , and generates new envmaps texture when an envmap is imported in an HDR database
 */
class EnvmapTextureManager final : private ISubscriber<database::event::ImageUpdateMessage *> {
  using Message = database::event::ImageUpdateMessage;

 public:
  explicit EnvmapTextureManager(ResourceDatabaseManager &resource_db,
                                Dim2 &screen_size,
                                unsigned int &default_framebuffer_id,
                                RenderPipeline &render_pipeline,
                                CubeMapMesh &skybox_mesh,
                                Scene &scene);

  ~EnvmapTextureManager() = default;
  EnvmapTextureManager(EnvmapTextureManager &copy) = delete;
  EnvmapTextureManager(EnvmapTextureManager &&move) = delete;
  EnvmapTextureManager &operator=(EnvmapTextureManager &copy) = delete;
  EnvmapTextureManager &operator=(EnvmapTextureManager &&move) = delete;

  void initializeLUT();
  void notified(observer::Data<Message *> &message) override;

  [[nodiscard]] int currentCubemapId() const { return current.cubemap_id; }
  [[nodiscard]] int currentPrefilterId() const { return current.prefiltered_id; }
  [[nodiscard]] int currentIrradianceId() const { return current.irradiance_id; }
  [[nodiscard]] int currentLutId() const { return current.lut_id; }

  void next();
  void previous();
  void updateCurrent(int index);

 private:
  void deleteFromCollection(int index);
  void addToCollection(int index);

 private:
  ResourceDatabaseManager &resource_database;
  std::vector<texture::envmap::EnvmapTextureGroup> bakes_id;
  texture::envmap::EnvmapTextureGroup current{};
  texture::envmap::EnvmapBakingConfig config{};
  unsigned current_counter{0};
  Dim2 &screen_dim;
  bool cuda_process; /* Use cuda for baking*/
  unsigned int &default_framebuffer_id;
  RenderPipeline &render_pipeline;
  CubeMapMesh &skybox_mesh; /*Mesh that will be operated on by this class*/
  Scene &scene;
};

#endif
