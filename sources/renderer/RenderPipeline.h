#ifndef RENDERPIPELINE_H
#define RENDERPIPELINE_H

#include "RenderCubeMap.h"
#include "RenderQuad.h"
#include "ResourceDatabaseManager.h"
#include "Scene.h"
/**
 * @file RenderPipeline.h
 * This file implements the rendering steps for each techniques , like deferred rendering , depth peeling , texture
 * baking  , occlusion culling etc.
 */
class GLViewer;
class Renderer;

/**
 * @class RenderPipeline
 */
class RenderPipeline : public controller::IProgressManager {
 public:
  explicit RenderPipeline(unsigned int &default_framebuffer_id, GLViewer &context, ResourceDatabaseManager *resource_database = nullptr);

  /**
   * @brief This method will bake an Environment map into a cubemap.
   * A new cubemap texture will be created , and stored in the texture database.
   * In addition , the texture will be assigned to the cubemap mesh.
   *
   * @param hdri_map  EnvironmentMapTexture that will be baked into a cubemap
   * @param cubemap_mesh The main scene's cubemap mesh .
   * @param width Width of the texture baked
   * @param height Height of the texture baked
   * @return Cubemap texture id
   */
  int bakeEnvmapToCubemap(EnvironmentMap2DTexture *hdri_map, CubeMapMesh &cubemap_mesh, unsigned width, unsigned height, Dim2 default_dim);

  /**
   * @brief This method produces an irradiance texture , that it will store inside the texture database
   *
   * @param cube_envmap Pre-computed Cubemap database index , from an environment map
   * @param width Width of the irradiance texture
   * @param height Height of the irradiance texture
   * @return int Irradiance texture database index
   */
  int bakeIrradianceCubemap(int cube_envmap, unsigned width, unsigned height, Dim2 default_dim);

  /**
   * @brief
   *
   * @return * void
   */
  virtual void clean();

  /**
   * @brief Generates mip maps of the environment map according to a computed roughness
   *
   * @param cube_envmap_database_id Cubemap of the environment map database ID
   * @param width
   * @param height
   * @param mipmap_levels Mip maps level
   * @param base_sample Base amount of sampling .
   * @param factor_per_mip Factor of sampling per mip levels ... more samples for high roughness
   * @return int Database ID of the mip mapped cubemap with roughness levels
   */
  int preFilterEnvmap(int cube_envmap_database_id,
                      unsigned int resolution,
                      unsigned int width,
                      unsigned int height,
                      unsigned int mipmap_levels,
                      unsigned int base_samples,
                      unsigned int factor_per_mip,
                      Dim2 default_dim);

  int generateBRDFLookupTexture(unsigned int width, unsigned int height, Dim2 default_dim);

 protected:
  /**
   * @brief Constructs a quad mesh , wrap it inside a drawable , and returns it
   *
   * @param shader
   * @param camera
   * @return Drawable
   */
  Drawable constructQuad(Shader *shader, Camera *camera);

  /**
   * @brief Construct a cube mesh , wrap it inside a drawable , and returns the drawable
   *
   * @param shader
   * @param database_texture_id
   * @param type
   * @param camera
   * @return Drawable
   */
  Drawable constructCube(Shader *shader, int database_texture_id, Texture::TYPE type, Camera *camera);

  /**
   * @brief Constructs a framebuffer object that will render to a cubemap
   *
   * @param dimensions Dimensions of the texture cubemap
   * @param persistence Texture database persistence . True if needs to be kept between scenes.
   * @param color_attachment Color attachment to use . <- temporary parameter.
   * @param internal_format Internal format of the cubemap
   * @param data_format Data format of the cubemap
   * @param data_type Data type of the cubemap
   * @param texture_type Type of the cubemap
   * @param shader Shader to initialize the texture with
   * @param level Cubemap mipmap level
   * @return RenderCubeMap Constructed FBO
   */
  template<class TEXTYPE>
  RenderCubeMap constructCubemapFbo(Dim2 *dimensions,
                                    bool persistence,
                                    GLFrameBuffer::INTERNAL_FORMAT color_attachment,
                                    Texture::FORMAT internal_format,
                                    Texture::FORMAT data_format,
                                    Texture::FORMAT data_type,
                                    Shader *shader,
                                    unsigned level = 0);
  template<class TEXTYPE>
  RenderQuadFBO constructQuadFbo(Dim2 *dimensions,
                                 bool persistence,
                                 GLFrameBuffer::INTERNAL_FORMAT color_attachment,
                                 Texture::FORMAT internal_format,
                                 Texture::FORMAT data_format,
                                 Texture::FORMAT data_type,
                                 Shader *shader);
  /**
   * @brief
   *
   * @param cube_drawable
   * @param cubemap_framebuffer
   * @param camera
   * @param render_viewport
   * @param origin_viewport
   * @param mip_level
   */
  void renderToCubemap(Drawable &cube_drawable,
                       RenderCubeMap &cubemap_framebuffer,
                       Camera &camera,
                       Dim2 render_viewport,
                       Dim2 origin_viewport,
                       unsigned int mip_level = 0);

  /**
   * @brief
   *
   * @param quad_drawable
   * @param quad_framebuffer
   * @param camera
   * @param render_viewport
   * @param origin_viewport
   * @param mip_level
   */
  void renderToQuad(Drawable &quad_drawable,
                    RenderQuadFBO &quad_framebuffer,
                    Camera &camera,
                    Dim2 render_viewport,
                    Dim2 origin_viewport,
                    unsigned int mip_level = 0);

 private:
  unsigned int &default_framebuffer_id;
  ResourceDatabaseManager *resource_database;
  GLViewer &context;
};

#endif