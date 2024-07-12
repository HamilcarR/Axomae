#include "RenderPipeline.h"
#include "DebugGL.h"
#include "FramebufferHelper.h"
#include "FreePerspectiveCamera.h"
#include "INodeFactory.h"
#include "RenderCubeMap.h"
#include "RenderQuad.h"
#include "Renderer.h"
#include "ShaderDatabase.h"

const float FOV = FOV;
const float NEAR = NEAR;
const float FAR = FAR;
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

RenderPipeline::RenderPipeline(unsigned int *default_framebuffer_id_, GLViewer *context_, ResourceDatabaseManager *_resource_database)
    : default_framebuffer_id(default_framebuffer_id_), resource_database(_resource_database), context(context_) {}

void RenderPipeline::clean() {}

int RenderPipeline::bakeEnvmapToCubemap(EnvironmentMap2DTexture *hdri_map, CubeMapMesh &cubemap, unsigned width, unsigned height, Dim2 default_dim_) {
  LOG("Generating an environment cubemap", LogLevel::INFO);
  AX_ASSERT(hdri_map != nullptr, "Texture is invalid.");
  AX_ASSERT(resource_database != nullptr, "Resource database is invalid");
  TextureDatabase &texture_database = *resource_database->getTextureDatabase();
  AX_ASSERT(!texture_database.empty(), "Texture database is empty.");

  auto progress = controller::progress_bar::generateData("Generating cubemap texture", 0);
  progress_manager->op(&progress);
  context->makeCurrent();

  ShaderDatabase &shader_database = *resource_database->getShaderDatabase();
  Shader *bake_shader = shader_database.get(Shader::ENVMAP_CUBEMAP_CONVERTER);
  Dim2 tex_dim{width, height};
  RenderCubeMap cubemap_renderer_framebuffer = constructCubemapFbo<CubemapTexture>(
      &tex_dim, false, GLFrameBuffer::COLOR0, Texture::RGB32F, Texture::RGB, Texture::FLOAT, bake_shader);
  auto query_envmap_result = texture_database.contains(hdri_map);
  int database_id_envmap = query_envmap_result.id;
  Dim2 cam_dim{width, height};
  FreePerspectiveCamera camera(FOV, &cam_dim, NEAR, FAR);  // Generic camera
  Drawable cube_drawable = constructCube(bake_shader, database_id_envmap, Texture::ENVMAP2D, &camera);
  Dim2 default_dim{default_dim_.width, default_dim_.height};
  renderToCubemap(cube_drawable, cubemap_renderer_framebuffer, camera, tex_dim, default_dim);
  cube_drawable.clean();
  cubemap_renderer_framebuffer.clean();

  auto query_baked_cubemap_texture = texture_database.contains(cubemap_renderer_framebuffer.getFrameBufferTexturePointer(GLFrameBuffer::COLOR0));
  query_baked_cubemap_texture.object->generateMipmap();
  errorCheck(__FILE__, __LINE__);
  progress_manager->reset();
  return query_baked_cubemap_texture.id;
}

int RenderPipeline::bakeIrradianceCubemap(int cube_envmap, unsigned width, unsigned height, Dim2 default_dim_) {
  LOG("Generating an irradiance cubemap", LogLevel::INFO);
  AX_ASSERT(context != nullptr, "GLViewer context is not valid.");

  auto progress = controller::progress_bar::generateData("Generating diffuse environment map", 0);
  progress_manager->op(&progress);
  context->makeCurrent();

  Dim2 irrad_dim{width, height};
  ShaderDatabase *shader_database = resource_database->getShaderDatabase();
  Shader *irradiance_shader = shader_database->get(Shader::IRRADIANCE_CUBEMAP_COMPUTE);
  RenderCubeMap cubemap_irradiance_framebuffer = constructCubemapFbo<IrradianceTexture>(
      &irrad_dim, false, GLFrameBuffer::COLOR0, Texture::RGB32F, Texture::RGB, Texture::FLOAT, irradiance_shader);
  Dim2 cam_dim{width, height};
  FreePerspectiveCamera camera(FOV, &cam_dim, NEAR, FAR);
  Drawable cube_drawable = constructCube(irradiance_shader, cube_envmap, Texture::CUBEMAP, &camera);
  Dim2 default_dim{default_dim_.width, default_dim_.height};
  renderToCubemap(cube_drawable, cubemap_irradiance_framebuffer, camera, irrad_dim, default_dim);
  cube_drawable.clean();
  cubemap_irradiance_framebuffer.clean();

  TextureDatabase *texture_database = resource_database->getTextureDatabase();
  auto query_baked_irradiance_texture = texture_database->contains(
      cubemap_irradiance_framebuffer.getFrameBufferTexturePointer(GLFrameBuffer::COLOR0));
  AX_ASSERT(query_baked_irradiance_texture.object != nullptr, "Irradiance texture is not valid.");
  errorCheck(__FILE__, __LINE__);
  progress_manager->reset();
  return query_baked_irradiance_texture.id;
}

int RenderPipeline::preFilterEnvmap(int cube_envmap,
                                    unsigned int resolution,
                                    unsigned int width,
                                    unsigned int height,
                                    unsigned int max_mip_level,
                                    unsigned int base_sample_count,
                                    unsigned int factor_per_mip,
                                    Dim2 default_dim_) {
  LOG("Generating a prefiltered cubemap", LogLevel::INFO);
  auto progress = controller::progress_bar::generateData("Generating specular environment map", 0);
  progress_manager->op(&progress);
  context->makeCurrent();

  Dim2 cubemap_dim{width, height};
  FreePerspectiveCamera camera(FOV, &cubemap_dim, NEAR, FAR);

  Dim2 resize_dim{width, height};
  ShaderDatabase *shader_database = resource_database->getShaderDatabase();
  auto *prefilter_shader = dynamic_cast<EnvmapPrefilterBakerShader *>(shader_database->get(Shader::ENVMAP_PREFILTER));
  RenderCubeMap &&cubemap_prefilter_fbo = constructCubemapFbo<CubemapTexture>(
      &resize_dim, false, GLFrameBuffer::COLOR0, Texture::RGB32F, Texture::RGB, Texture::FLOAT, prefilter_shader, max_mip_level);

  cubemap_prefilter_fbo.getFrameBufferTexturePointer(GLFrameBuffer::COLOR0)->generateMipmap();
  cubemap_prefilter_fbo.bindFrameBuffer();
  Drawable cube_drawable = constructCube(prefilter_shader, cube_envmap, Texture::CUBEMAP, &camera);
  cube_drawable.startDraw();
  cube_drawable.bind();

  /* Computes multiple specular irradiance mipmaps  , based on pre-determined roughness values*/
  for (unsigned i = 0; i < max_mip_level; i++) {
    resize_dim.width = (int)(cubemap_dim.width / std::pow(2.f, i));
    resize_dim.height = (int)(cubemap_dim.height / std::pow(2.f, i));
    cubemap_prefilter_fbo.resize();
    float roughness = (float)i / (float)(max_mip_level - 1);
    prefilter_shader->setRoughnessValue(roughness);
    prefilter_shader->setSamplesCount(base_sample_count * i * factor_per_mip);
    prefilter_shader->setCubeEnvmapResolution(resolution);
    glViewport(0, 0, (int)resize_dim.width, (int)resize_dim.height);
    for (unsigned k = 0; k < 6; k++) {
      camera.setView(views[k]);
      prefilter_shader->setAllMatricesUniforms(glm::mat4(1.f));
      cubemap_prefilter_fbo.renderToTexture(k, GLFrameBuffer::COLOR0, i);
      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
      Mesh *cube_draw_mesh_ptr = cube_drawable.getMeshPointer();
      glDrawElements(GL_TRIANGLES, (int)cube_draw_mesh_ptr->getGeometry().indices.size(), GL_UNSIGNED_INT, nullptr);
      glFinish();
    }
  }

  Dim2 default_dim = default_dim_;
  glViewport(0, 0, (int)default_dim.width, (int)default_dim.height);
  cube_drawable.unbind();
  cube_drawable.clean();
  cubemap_prefilter_fbo.unbindFrameBuffer();
  cubemap_prefilter_fbo.clean();

  TextureDatabase *texture_database = resource_database->getTextureDatabase();
  database::Result<int, Texture> query_prefiltered_cubemap_texture = texture_database->contains(
      cubemap_prefilter_fbo.getFrameBufferTexturePointer(GLFrameBuffer::COLOR0));

  AX_ASSERT(query_prefiltered_cubemap_texture.object, "");
  progress_manager->reset();
  errorCheck(__FILE__, __LINE__);
  return query_prefiltered_cubemap_texture.id;
}

int RenderPipeline::generateBRDFLookupTexture(unsigned int width, unsigned int height, Dim2 default_dim_) {

  auto pbar_status = controller::progress_bar::generateData("Generating BRDF look up table", 0);
  progress_manager->op(&pbar_status);
  context->makeCurrent();

  auto *brdf_lut_shader = dynamic_cast<BRDFLookupTableBakerShader *>(resource_database->getShaderDatabase()->get(Shader::BRDF_LUT_BAKER));
  Dim2 camera_dim{1, 1};
  FreePerspectiveCamera camera(FOV, &camera_dim, NEAR, FAR);
  Dim2 tex_dim{width, height};
  RenderQuadFBO quad_fbo = constructQuadFbo<BRDFLookupTexture>(
      &tex_dim, false, GLFrameBuffer::COLOR0, Texture::RGB16F, Texture::RG, Texture::FLOAT, brdf_lut_shader);
  Drawable quad_drawable = constructQuad(brdf_lut_shader, &camera);
  Dim2 original_dim = default_dim_;
  renderToQuad(quad_drawable, quad_fbo, camera, tex_dim, original_dim);
  quad_drawable.clean();
  quad_fbo.clean();

  database::Result<int, Texture> query_brdf_lut = resource_database->getTextureDatabase()->contains(
      quad_fbo.getFrameBufferTexturePointer(GLFrameBuffer::COLOR0));
  errorCheck(__FILE__, __LINE__);
  AX_ASSERT(query_brdf_lut.object, "");
  progress_manager->reset();
  return query_brdf_lut.id;
}

void RenderPipeline::renderToCubemap(Drawable &cube_drawable,
                                     RenderCubeMap &cubemap_framebuffer,
                                     Camera &camera,
                                     const Dim2 render_viewport,
                                     const Dim2 origin_viewport,
                                     unsigned int mip_level) {
  cubemap_framebuffer.bindFrameBuffer();
  cube_drawable.startDraw();
  glViewport(0, 0, (int)render_viewport.width, (int)render_viewport.height);
  for (unsigned i = 0; i < 6; i++) {
    camera.setView(views[i]);
    cube_drawable.bind();
    cubemap_framebuffer.renderToTexture(i, GLFrameBuffer::COLOR0, mip_level);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    Mesh *cube_draw_mesh_ptr = cube_drawable.getMeshPointer();
    glDrawElements(GL_TRIANGLES, (int)cube_draw_mesh_ptr->getGeometry().indices.size(), GL_UNSIGNED_INT, nullptr);
  }
  cube_drawable.unbind();
  cubemap_framebuffer.unbindFrameBuffer();
  glViewport(0, 0, (int)origin_viewport.width, (int)origin_viewport.height);
}

void RenderPipeline::renderToQuad(Drawable &quad_drawable,
                                  RenderQuadFBO &quad_framebuffer,
                                  Camera &camera,
                                  const Dim2 render_viewport,
                                  const Dim2 origin_viewport,
                                  unsigned int mip_level) {
  quad_framebuffer.bindFrameBuffer();
  quad_drawable.startDraw();
  glViewport(0, 0, (int)render_viewport.width, (int)render_viewport.height);
  quad_drawable.bind();
  quad_framebuffer.renderToTexture(GLFrameBuffer::COLOR0);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  Mesh *quad_draw_mesh_ptr = quad_drawable.getMeshPointer();
  glDrawElements(GL_TRIANGLES, (int)quad_draw_mesh_ptr->getGeometry().indices.size(), GL_UNSIGNED_INT, nullptr);
  quad_drawable.unbind();
  quad_framebuffer.unbindFrameBuffer();
  glViewport(0, 0, (int)origin_viewport.width, (int)origin_viewport.height);
}

Drawable RenderPipeline::constructCube(Shader *shader, int database_texture_id, Texture::TYPE type, Camera *camera) {
  assert(shader != nullptr);
  CubeMesh *cube = database::node::store<CubeMesh>(*resource_database->getNodeDatabase(), false).object;
  cube->setShader(shader);
  MaterialInterface *material = cube->getMaterial();
  material->addTexture(database_texture_id);
  cube->setSceneCameraPointer(camera);
  return Drawable(cube);
}

Drawable RenderPipeline::constructQuad(Shader *shader, Camera *camera) {
  assert(shader != nullptr);
  QuadMesh *quad = database::node::store<QuadMesh>(*resource_database->getNodeDatabase(), false).object;
  quad->setShader(shader);
  quad->setSceneCameraPointer(camera);
  return Drawable(quad);
}

template<class TEXTYPE>
RenderCubeMap RenderPipeline::constructCubemapFbo(Dim2 *dimensions,
                                                  bool persistence,
                                                  GLFrameBuffer::INTERNAL_FORMAT color_attachment,
                                                  Texture::FORMAT internal_format,
                                                  Texture::FORMAT data_format,
                                                  Texture::FORMAT data_type,
                                                  Shader *shader,
                                                  unsigned int mipmaps_level) {

  TextureDatabase *texture_database = resource_database->getTextureDatabase();
  RenderCubeMap cubemap_fbo(texture_database, dimensions, default_framebuffer_id);
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
RenderQuadFBO RenderPipeline::constructQuadFbo(Dim2 *dimensions,
                                               bool persistence,
                                               GLFrameBuffer::INTERNAL_FORMAT color_attachment,
                                               Texture::FORMAT internal_format,
                                               Texture::FORMAT data_format,
                                               Texture::FORMAT data_type,
                                               Shader *shader) {

  TextureDatabase *texture_database = resource_database->getTextureDatabase();
  RenderQuadFBO quad_fbo(texture_database, dimensions, default_framebuffer_id);
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