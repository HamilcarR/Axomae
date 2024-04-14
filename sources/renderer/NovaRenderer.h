#ifndef NOVARENDERER_H
#define NOVARENDERER_H

#include "LightingDatabase.h"
#include "RendererInterface.h"

class EnvmapTextureManager;
class GLViewer;
class CameraFrameBuffer;
class ResourceDatabaseManager;
class RenderPipeline;
class GLPixelBufferObject;
class EnvmapTextureManager;

class NovaRenderer final : public IRenderer {

 public:
  std::unique_ptr<RenderPipeline> render_pipeline;
  std::unique_ptr<CameraFrameBuffer> camera_framebuffer;
  bool start_draw{};
  ResourceDatabaseManager *resource_database{};
  std::unique_ptr<Scene> scene;
  Camera *scene_camera{};
  Dim2 screen_size{};
  unsigned int default_framebuffer_id{};
  LightingDatabase light_database;
  GLViewer *gl_widget{};

 private:
  Texture *framebuffer_texture{};
  std::unique_ptr<GLPixelBufferObject> pixel_buffer_object;
  std::unique_ptr<EnvmapTextureManager> envmap_manager;

 private:
  NovaRenderer() = default;

 public:
  NovaRenderer(unsigned width, unsigned height, GLViewer *widget = nullptr);
  ~NovaRenderer() override;
  NovaRenderer(const NovaRenderer &copy) = delete;
  NovaRenderer &operator=(const NovaRenderer &copy) = delete;
  NovaRenderer(NovaRenderer &&move) noexcept = default;
  NovaRenderer &operator=(NovaRenderer &&move) noexcept = default;
  void initialize(ApplicationConfig *app_conf) override;
  bool prep_draw() override;
  void draw() override;
  void processEvent(const controller::event::Event *event) override;
  void onResize(unsigned int width, unsigned int height) override;
  void setDefaultFrameBufferId(unsigned id) override;
  void setNewScene(const SceneChangeData &new_scene) override;
  [[nodiscard]] unsigned int *getDefaultFrameBufferIdPointer() override;
  [[nodiscard]] RenderPipeline &getRenderPipeline() const override;
  [[nodiscard]] Scene &getScene() const override;

  void setGammaValue(float gamma) override;
  void setExposureValue(float exposure) override;
  void setNoPostProcess() override;
  void setPostProcessEdge() override;
  void setPostProcessSharpen() override;
  void setPostProcessBlurr() override;
  void resetSceneCamera() override;
  void setRasterizerPoint() override;
  void setRasterizerFill() override;
  void setRasterizerWireframe() override;
  void displayBoundingBoxes(bool display) override;

  void setViewerWidget(GLViewer *widget) override;
};

#endif  // NOVARENDERER_H
