#include "RenderPipeline.h"
#include "DebugGL.h"
#include "FrameBufferInterface.h"
#include "INodeFactory.h"
#include "RenderCubeMap.h"
#include "RenderQuad.h"
#include "Renderer.h"
#include <QOffscreenSurface>

namespace camera_angles {
  const auto up = glm::lookAt(glm::vec3(0.f), glm::vec3(0.f, 1.f, 0.f), glm::vec3(0.f, 0.f, -1.f));
  const auto down = glm::lookAt(glm::vec3(0.f), glm::vec3(0.f, -1.f, 0.f), glm::vec3(0.f, 0.f, 1.f));
  const auto left = glm::lookAt(glm::vec3(0.f), glm::vec3(-1.f, 0.f, 0.f), glm::vec3(0.f, 1.f, 0.f));
  const auto right = glm::lookAt(glm::vec3(0.f), glm::vec3(1.f, 0.f, 0.f), glm::vec3(0.f, 1.f, 0.f));
  const auto back = glm::lookAt(glm::vec3(0.f), glm::vec3(0.f, 0.f, 1.f), glm::vec3(0.f, 1.f, 0.f));
  const auto front = glm::lookAt(glm::vec3(0.f), glm::vec3(0.f, 0.f, -1.f), glm::vec3(0.f, 1.f, 0.f));
}  // namespace camera_angles
const std::vector<glm::mat4> views = {
    camera_angles::left, camera_angles::right, camera_angles::down, camera_angles::up, camera_angles::back, camera_angles::front};

RenderPipeline::RenderPipeline(Renderer *_renderer, ResourceDatabaseManager *_resource_database) {
  renderer = _renderer;
  resource_database = _resource_database;
}

RenderPipeline::~RenderPipeline() {}

void RenderPipeline::clean() {}
/********************************************************************************************************************************************************************************************************/
/**
 * The function "bakeEnvmapToCubemap" takes an environment map HDR texture and converts it into a floating point
 * cubemap.
 *
 * @param hdri_map The `hdri_map` parameter is an `EnvironmentMapTexture` object, which represents the
 * high dynamic range image used for the environment map. It contains the texture data and other
 * properties of the image.
 * @param width The width of the cubemap texture to be rendered.
 * @param height The height parameter in the code represents the height of the cubemap texture that
 * will be generated. It determines the resolution of the cubemap texture.
 * @param gl_widget GLViewer object used to extract informations like default FBO id , current width/height of the
 * rendering surface etc.
 *
 * @return a pointer to a CubeMapMesh object.
 */
CubeMapMesh *RenderPipeline::bakeEnvmapToCubemap(EnvironmentMap2DTexture *hdri_map, unsigned width, unsigned height, GLViewer *gl_widget) {
  LOG("Generating an environment cubemap", LogLevel::INFO);
  assert(resource_database != nullptr);
  assert(!resource_database->getTextureDatabase().empty());
  assert(renderer != nullptr);
  assert(hdri_map != nullptr);
  TextureDatabase *texture_database = &resource_database->getTextureDatabase();
  ShaderDatabase *shader_database = &resource_database->getShaderDatabase();
  Shader *bake_shader = shader_database->get(Shader::ENVMAP_CUBEMAP_CONVERTER);
  ScreenSize tex_dim, cam_dim, default_dim;
  /*Set up camera ratio + cubemap texture resolution*/
  default_dim.width = gl_widget->width();
  default_dim.height = gl_widget->height();
  cam_dim.width = width;
  cam_dim.height = height;
  tex_dim.width = width;
  tex_dim.height = height;
  FreePerspectiveCamera camera(90.f, &cam_dim, 0.1f, 2000.f);  // Generic camera
  database::Result<int, Texture> query_envmap_result = texture_database->contains(hdri_map);
  int database_id_envmap = query_envmap_result.id;
  /* Generate a framebuffer that will render to a cubemap*/
  RenderCubeMap cubemap_renderer_framebuffer = constructCubemapFbo<CubemapTexture>(
      &tex_dim, false, GLFrameBuffer::COLOR0, Texture::RGB32F, Texture::RGB, Texture::FLOAT, bake_shader);
  Drawable cube_drawable = constructCube(bake_shader, database_id_envmap, Texture::ENVMAP2D, &camera);
  renderToCubemap(cube_drawable, cubemap_renderer_framebuffer, camera, tex_dim, default_dim);
  cube_drawable.clean();
  cubemap_renderer_framebuffer.clean();
  database::Result<int, Texture> query_baked_cubemap_texture = texture_database->contains(
      cubemap_renderer_framebuffer.getFrameBufferTexturePointer(GLFrameBuffer::COLOR0));
  /* Mesh to be returned */
  CubeMapMesh *cubemap = database::node::store<CubeMapMesh>(resource_database->getNodeDatabase(), false).object;
  query_baked_cubemap_texture.object->generateMipmap();
  cubemap->material.addTexture(query_baked_cubemap_texture.id);
  cubemap->setShader(shader_database->get(Shader::CUBEMAP));
  errorCheck(__FILE__, __LINE__);
  texture_database->remove(hdri_map);
  return cubemap;
}

/********************************************************************************************************************************************************************************************************/
/**
 * The function `bakeIrradianceCubemap` renders a cube map by computing the irradiance values for each
 * texel using a given environment map.
 *
 * @param cube_envmap The cube_envmap parameter is the ID of the cube environment map texture that will
 * be used for baking the irradiance cubemap.
 * @param width The width of the output irradiance cubemap in pixels.
 * @param height The "height" parameter in the code represents the height of the output irradiance
 * cubemap texture.
 * @param gl_widget GLViewer object that represents the OpenGL widget used for rendering.
 *
 * @return an integer value, which is the database ID of the baked irradiance cubemap texture.
 */
int RenderPipeline::bakeIrradianceCubemap(int cube_envmap, unsigned width, unsigned height, GLViewer *gl_widget) {
  LOG("Generating an irradiance cubemap", LogLevel::INFO);
  ScreenSize irrad_dim, cam_dim, default_dim;
  TextureDatabase *texture_database = &resource_database->getTextureDatabase();
  ShaderDatabase *shader_database = &resource_database->getShaderDatabase();
  Shader *irradiance_shader = shader_database->get(Shader::IRRADIANCE_CUBEMAP_COMPUTE);
  FreePerspectiveCamera camera(90.f, &cam_dim, 0.1f, 2000.f);
  cam_dim.width = width;
  cam_dim.height = height;
  irrad_dim.width = width;
  irrad_dim.height = height;
  default_dim.width = gl_widget->width();
  default_dim.height = gl_widget->height();
  RenderCubeMap cubemap_irradiance_framebuffer = constructCubemapFbo<IrradianceTexture>(
      &irrad_dim, false, GLFrameBuffer::COLOR0, Texture::RGB32F, Texture::RGB, Texture::FLOAT, irradiance_shader);
  Drawable cube_drawable = constructCube(irradiance_shader, cube_envmap, Texture::CUBEMAP, &camera);
  renderToCubemap(cube_drawable, cubemap_irradiance_framebuffer, camera, irrad_dim, default_dim);
  cube_drawable.clean();
  cubemap_irradiance_framebuffer.clean();
  database::Result<int, Texture> query_baked_irradiance_texture = texture_database->contains(
      cubemap_irradiance_framebuffer.getFrameBufferTexturePointer(GLFrameBuffer::COLOR0));
  assert(query_baked_irradiance_texture.object);
  errorCheck(__FILE__, __LINE__);
  return query_baked_irradiance_texture.id;
}
/********************************************************************************************************************************************************************************************************/

int RenderPipeline::preFilterEnvmap(int cube_envmap,
                                    unsigned int resolution,
                                    unsigned int width,
                                    unsigned int height,
                                    unsigned int max_mip_level,
                                    unsigned int base_sample_count,
                                    unsigned int factor_per_mip,
                                    GLViewer *gl_widget) {
  LOG("Generating a prefiltered cubemap", LogLevel::INFO);
  ScreenSize cubemap_dim, default_dim, resize_dim;
  EnvmapPrefilterBakerShader *prefilter_shader = static_cast<EnvmapPrefilterBakerShader *>(
      resource_database->getShaderDatabase().get(Shader::ENVMAP_PREFILTER));
  auto texture_database = &resource_database->getTextureDatabase();
  cubemap_dim.width = width;
  cubemap_dim.height = height;
  resize_dim = cubemap_dim;
  default_dim.width = gl_widget->width();
  default_dim.height = gl_widget->height();
  FreePerspectiveCamera camera(90.f, &cubemap_dim, 0.1f, 2000.f);
  RenderCubeMap &&cubemap_prefilter_fbo = constructCubemapFbo<CubemapTexture>(
      &resize_dim, false, GLFrameBuffer::COLOR0, Texture::RGB32F, Texture::RGB, Texture::FLOAT, prefilter_shader, max_mip_level);
  Drawable cube_drawable = constructCube(prefilter_shader, cube_envmap, Texture::CUBEMAP, &camera);
  cubemap_prefilter_fbo.getFrameBufferTexturePointer(GLFrameBuffer::COLOR0)->generateMipmap();
  cubemap_prefilter_fbo.bindFrameBuffer();
  cube_drawable.startDraw();
  cube_drawable.bind();
  for (unsigned i = 0; i < max_mip_level; i++) {
    resize_dim.width = cubemap_dim.width / std::pow(2.f, i);
    resize_dim.height = cubemap_dim.height / std::pow(2.f, i);
    cubemap_prefilter_fbo.resize();
    float roughness = (float)i / (float)(max_mip_level - 1);
    prefilter_shader->setRoughnessValue(roughness);
    prefilter_shader->setSamplesCount(base_sample_count * i * factor_per_mip);
    prefilter_shader->setCubeEnvmapResolution(resolution);
    glViewport(0, 0, resize_dim.width, resize_dim.height);
    for (unsigned k = 0; k < 6; k++) {
      camera.setView(views[k]);
      prefilter_shader->setAllMatricesUniforms(glm::mat4(1.f));
      cubemap_prefilter_fbo.renderToTexture(k, GLFrameBuffer::COLOR0, i);
      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
      glDrawElements(GL_TRIANGLES, cube_drawable.getMeshPointer()->geometry.indices.size(), GL_UNSIGNED_INT, 0);
      glFinish();
    }
  }

  glViewport(0, 0, default_dim.width, default_dim.height);
  cube_drawable.unbind();
  cube_drawable.clean();
  cubemap_prefilter_fbo.unbindFrameBuffer();
  cubemap_prefilter_fbo.clean();
  database::Result<int, Texture> query_prefiltered_cubemap_texture = texture_database->contains(
      cubemap_prefilter_fbo.getFrameBufferTexturePointer(GLFrameBuffer::COLOR0));
  errorCheck(__FILE__, __LINE__);
  assert(query_prefiltered_cubemap_texture.object);
  return query_prefiltered_cubemap_texture.id;
}

/********************************************************************************************************************************************************************************************************/

int RenderPipeline::generateBRDFLookupTexture(unsigned int width, unsigned int height, GLViewer *gl_widget) {
  ScreenSize tex_dim, camera_dim, original_dim;
  tex_dim.width = width;
  tex_dim.height = height;
  camera_dim.width = 1;
  camera_dim.height = 1;
  original_dim.width = gl_widget->width();
  original_dim.height = gl_widget->height();
  BRDFLookupTableBakerShader *brdf_lut_shader = static_cast<BRDFLookupTableBakerShader *>(
      resource_database->getShaderDatabase().get(Shader::BRDF_LUT_BAKER));
  FreePerspectiveCamera camera(90.f, &camera_dim, 0.1f, 2000.f);
  RenderQuadFBO &&quad_fbo = constructQuadFbo<BRDFLookupTexture>(
      &tex_dim, false, GLFrameBuffer::COLOR0, Texture::RGB16F, Texture::RG, Texture::FLOAT, brdf_lut_shader);
  Drawable quad_drawable = constructQuad(brdf_lut_shader, &camera);
  renderToQuad(quad_drawable, quad_fbo, camera, tex_dim, original_dim);
  quad_drawable.clean();
  quad_fbo.clean();
  database::Result<int, Texture> query_brdf_lut = resource_database->getTextureDatabase().contains(
      quad_fbo.getFrameBufferTexturePointer(GLFrameBuffer::COLOR0));
  errorCheck(__FILE__, __LINE__);
  assert(query_brdf_lut.object);
  return query_brdf_lut.id;
}

/********************************************************************************************************************************************************************************************************/
/**
 * The function renders a given drawable object to a cubemap framebuffer using a specified camera and
 * viewport sizes.
 *
 * @param cube_drawable A reference to a Drawable object that represents the cube to be rendered.
 * @param cubemap_framebuffer The `cubemap_framebuffer` parameter is an object of type `RenderCubeMap`,
 * which represents the framebuffer used for rendering to a cubemap.
 * @param camera The camera object represents the viewpoint from which the scene will be rendered. It
 * contains information such as the camera position, orientation, and projection matrix.
 * @param render_viewport The render_viewport parameter is the size of the viewport where the cubemap
 * will be rendered. It specifies the width and height of the viewport in pixels.
 * @param origin_viewport The `origin_viewport` parameter represents the size of the original viewport
 * before rendering to the cubemap. It specifies the width and height of the viewport in pixels.
 */
void RenderPipeline::renderToCubemap(Drawable &cube_drawable,
                                     RenderCubeMap &cubemap_framebuffer,
                                     Camera &camera,
                                     const ScreenSize render_viewport,
                                     const ScreenSize origin_viewport,
                                     unsigned int mip_level) {
  cubemap_framebuffer.bindFrameBuffer();
  cube_drawable.startDraw();
  glViewport(0, 0, render_viewport.width, render_viewport.height);
  for (unsigned i = 0; i < 6; i++) {
    camera.setView(views[i]);
    cube_drawable.bind();
    cubemap_framebuffer.renderToTexture(i, GLFrameBuffer::COLOR0, mip_level);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glDrawElements(GL_TRIANGLES, cube_drawable.getMeshPointer()->geometry.indices.size(), GL_UNSIGNED_INT, 0);
  }
  cube_drawable.unbind();
  cubemap_framebuffer.unbindFrameBuffer();
  glViewport(0, 0, origin_viewport.width, origin_viewport.height);
}

void RenderPipeline::renderToQuad(Drawable &quad_drawable,
                                  RenderQuadFBO &quad_framebuffer,
                                  Camera &camera,
                                  const ScreenSize render_viewport,
                                  const ScreenSize origin_viewport,
                                  unsigned int mip_level) {
  quad_framebuffer.bindFrameBuffer();
  quad_drawable.startDraw();
  glViewport(0, 0, render_viewport.width, render_viewport.height);
  quad_drawable.bind();
  quad_framebuffer.renderToTexture(GLFrameBuffer::COLOR0);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glDrawElements(GL_TRIANGLES, quad_drawable.getMeshPointer()->geometry.indices.size(), GL_UNSIGNED_INT, 0);
  quad_drawable.unbind();
  quad_framebuffer.unbindFrameBuffer();
  glViewport(0, 0, origin_viewport.width, origin_viewport.height);
}

/********************************************************************************************************************************************************************************************************/
Drawable RenderPipeline::constructCube(Shader *shader, int database_texture_id, Texture::TYPE type, Camera *camera) {
  assert(shader != nullptr);
  CubeMesh *cube = database::node::store<CubeMesh>(resource_database->getNodeDatabase(), false).object;
  cube->setShader(shader);
  cube->material.addTexture(database_texture_id);
  cube->setSceneCameraPointer(camera);
  return Drawable(cube);
}

Drawable RenderPipeline::constructQuad(Shader *shader, Camera *camera) {
  assert(shader != nullptr);
  QuadMesh *quad = database::node::store<QuadMesh>(resource_database->getNodeDatabase(), false).object;
  quad->setShader(shader);
  quad->setSceneCameraPointer(camera);
  return Drawable(quad);
}

/********************************************************************************************************************************************************************************************************/
template<class TEXTYPE>
RenderCubeMap RenderPipeline::constructCubemapFbo(ScreenSize *dimensions,
                                                  bool persistence,
                                                  GLFrameBuffer::INTERNAL_FORMAT color_attachment,
                                                  Texture::FORMAT internal_format,
                                                  Texture::FORMAT data_format,
                                                  Texture::FORMAT data_type,
                                                  Shader *shader,
                                                  unsigned int mipmaps_level) {

  TextureDatabase *texture_database = &resource_database->getTextureDatabase();
  RenderCubeMap cubemap_fbo(texture_database, dimensions, renderer->getDefaultFrameBufferIdPointer());
  cubemap_fbo.initializeFrameBufferTexture<TEXTYPE>(
      color_attachment, persistence, internal_format, data_format, data_type, dimensions->width, dimensions->height, mipmaps_level);
  if (!shader->isInitialized())
    shader->initializeShader();
  shader->bind();
  auto fb_tpoint = cubemap_fbo.getFrameBufferTexturePointer(GLFrameBuffer::COLOR0);
  fb_tpoint->setGlData(shader);
  shader->release();
  cubemap_fbo.initializeFrameBuffer();
  errorCheck(__FILE__, __LINE__);
  return cubemap_fbo;
}

template<class TEXTYPE>
RenderQuadFBO RenderPipeline::constructQuadFbo(ScreenSize *dimensions,
                                               bool persistence,
                                               GLFrameBuffer::INTERNAL_FORMAT color_attachment,
                                               Texture::FORMAT internal_format,
                                               Texture::FORMAT data_format,
                                               Texture::FORMAT data_type,
                                               Shader *shader) {

  TextureDatabase *texture_database = &resource_database->getTextureDatabase();
  RenderQuadFBO quad_fbo(texture_database, dimensions, renderer->getDefaultFrameBufferIdPointer());
  quad_fbo.initializeFrameBufferTexture<TEXTYPE>(
      color_attachment, persistence, internal_format, data_format, data_type, dimensions->width, dimensions->height);
  if (!shader->isInitialized())
    shader->initializeShader();
  shader->bind();
  quad_fbo.getFrameBufferTexturePointer(GLFrameBuffer::COLOR0)->setGlData(shader);
  shader->release();
  quad_fbo.initializeFrameBuffer();
  errorCheck(__FILE__, __LINE__);
  return quad_fbo;
}