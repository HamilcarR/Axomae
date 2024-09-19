#ifndef ENVMAPTEXTUREMANAGER_H
#define ENVMAPTEXTUREMANAGER_H
#include "ImageDatabase.h"
#include "RenderPipeline.h"
#include "ResourceDatabaseManager.h"
#include "internal/macro/project_macros.h"
#include <internal/common/Observer.h>
#include <internal/common/axstd/span.h>

namespace texture::envmap {
  /*This structure packs all 3 cubemaps IDs used in IBL*/
  struct EnvmapTextureGroup {
    /*TextureDatabase IDs*/
    int cubemap_id;
    int prefiltered_id;
    int irradiance_id;
    int lut_id;
    /*ImageDatabase ID*/
    int equirect_id;
    /* GL ID */
    GLuint equirect_gl_id;
    GLuint cubemap_gl_id;
    GLuint irradiance_gl_id;
    GLuint lut_gl_id;
    GLuint prefiltered_gl_id;

    /*Raw data*/
    image::ImageHolder<float> *metadata;
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
    std::string default_envmap_path;
  };

}  // namespace texture::envmap

class ApplicationConfig;

/**
 * @class EnvmapTextureManager
 * @brief Tracks the current scene's envmap IDs , and generates new envmaps textures and IDs when an envmap is imported in an HDR database
 */
class EnvmapTextureManager final : private ISubscriber<database::event::ImageUpdateMessage *> {
  using Message = database::event::ImageUpdateMessage;

 public:
  /* How this EnvmapTextureManager instance will react to an envmap being imported*/
  enum UpdatePolicy {
    NONE = 0,
    ADD = 1 << 0,
    DELETE = 1 << 1,
    SELECTED = 1 << 2

  };

 private:
  ResourceDatabaseManager *resource_database;
  /* Different instances of the manager need to keep track of each others envmap generations*/
  static std::vector<texture::envmap::EnvmapTextureGroup> bakes_id;  // TODO: Abominable, refactor ASAP.
  texture::envmap::EnvmapTextureGroup current{};
  texture::envmap::EnvmapBakingConfig config{};
  unsigned current_counter{0};
  Dim2 screen_dim;
  bool cuda_process; /* Use cuda for baking*/
  unsigned int *default_framebuffer_id;
  RenderPipeline *render_pipeline;
  CubeMapMesh *skybox_mesh; /*skybox mesh that will be operated on by this class*/
  Scene *scene;
  int update_policy_flag;

 public:
  explicit EnvmapTextureManager(ResourceDatabaseManager &resource_db,
                                const Dim2 &screen_size,
                                unsigned int &default_framebuffer_id,
                                RenderPipeline &render_pipeline,
                                Scene *scene,
                                int policy = ADD | DELETE | SELECTED);

  ~EnvmapTextureManager() = default;

  EnvmapTextureManager(const EnvmapTextureManager &copy) = delete;

  EnvmapTextureManager(EnvmapTextureManager &&move) = delete;

  EnvmapTextureManager &operator=(const EnvmapTextureManager &copy) = delete;

  EnvmapTextureManager &operator=(EnvmapTextureManager &&move) = delete;

  void initializeDefaultEnvmap(ApplicationConfig *app_conf);

  void notified(observer::Data<Message *> &message) override;
  /* Returns the cubemap id from the TextureDatabase. */
  ax_no_discard int currentCubemapId() const { return current.cubemap_id; }

  /* Returns the prefiltered specular texture id from the TextureDatabase. */
  ax_no_discard int currentPrefilterId() const { return current.prefiltered_id; }

  /* Returns the convoluted irradiance texture  id from the TextureDatabase. */
  ax_no_discard int currentIrradianceId() const { return current.irradiance_id; }

  /* Returns the lut id from the TextureDatabase. */
  ax_no_discard int currentLutId() const { return current.lut_id; }

  /* Returns the equirectangular texture id from the TextureDatabase. */
  ax_no_discard int currentEquirect2D() const { return current.equirect_id; }

  ax_no_discard const texture::envmap::EnvmapTextureGroup &getCurrentEnvmapGroup() const { return current; }

  ax_no_discard const image::ImageHolder<float> *currentEnvmapMetadata() const { return current.metadata; }

  ax_no_discard image::ImageHolder<float> *currentMutableEnvmapMetadata() const { return current.metadata; }

  axstd::span<const texture::envmap::EnvmapTextureGroup> getBakesViews() const { return bakes_id; }

  std::size_t getEnvmapID() const;

  void next();

  void previous();

  void updateCurrent(int index);

 private:
  void deleteFromCollection(int index);

  void addToCollection(int index);
};

#endif
