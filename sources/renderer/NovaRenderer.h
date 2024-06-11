#ifndef NOVARENDERER_H
#define NOVARENDERER_H

#include "LightingDatabase.h"
#include "NovaInterface.h"
#include "RendererInterface.h"

namespace nova {
  struct NovaResources;
}
class EnvmapTextureManager;
class GLViewer;
class CameraFrameBuffer;
class ResourceDatabaseManager;
class RenderPipeline;
class GLMutablePixelBufferObject;
class EnvmapTextureManager;
class NovaRenderEngineInterface;

class NovaRenderer final : public IRenderer {

 private:
  struct NovaInternalMetadata {
    float max_channel_color_value;
  };

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
  std::unique_ptr<GLMutablePixelBufferObject> pbo_read;
  std::unique_ptr<NovaRenderEngineInterface> nova_engine;
  std::unique_ptr<EnvmapTextureManager> envmap_manager;
  std::unique_ptr<nova::NovaResources> nova_engine_data;
  ApplicationConfig *global_application_config{};
  std::vector<std::future<void>> nova_result_futures;
  std::vector<float> partial_render_buffer;
  std::vector<float> accumulated_render_buffer;
  std::vector<float> final_render_buffer;
  Dim2 resolution{2048, 2048};
  int current_frame{}, next_frame{}, scanline{};
  bool needRedraw{false};
  bool cancel_render{false};
  float *pbo_map_buffer{};
  NovaInternalMetadata renderer_data{};
  nova::HdrBufferStruct engine_render_buffers;

 private:
  NovaRenderer() = default;
  void updateNovaCameraFields();
  void copyBufferToPbo(float *pbo_mapped_buffer, int width, int height, int channels);
  void initializeEngine();
  void prepareRedraw();
  void emptyBuffers();
  void emptyAccumBuffer();
  void emptyRenderBuffer();
  void resetToBaseState();
  void displayProgress(float current, float target);

 public:
  NovaRenderer(unsigned width, unsigned height, GLViewer *widget = nullptr);
  ~NovaRenderer() override;
  NovaRenderer(const NovaRenderer &copy) = delete;
  NovaRenderer &operator=(const NovaRenderer &copy) = delete;
  NovaRenderer(NovaRenderer &&move) noexcept = default;
  NovaRenderer &operator=(NovaRenderer &&move) noexcept = default;
  void syncRenderEngineThreads();
  void getScreenPixelColor(int x, int y, float r_screen_pixel_color[4]) override;
  void populateNovaSceneResources();
  void initialize(ApplicationConfig *app_conf) override;
  bool prep_draw() override;
  void draw() override;
  void processEvent(const controller::event::Event *event) override;
  void onResize(unsigned int width, unsigned int height) override;
  void onClose() override;
  void setDefaultFrameBufferId(unsigned id) override;
  void setNewScene(const SceneChangeData &new_scene) override;
  [[nodiscard]] unsigned int *getDefaultFrameBufferIdPointer() override;
  [[nodiscard]] RenderPipeline &getRenderPipeline() const override;
  [[nodiscard]] Scene &getScene() const override;
  [[nodiscard]] image::ImageHolder<float> getSnapshotFloat(int width, int height) const override;
  [[nodiscard]] image::ImageHolder<uint8_t> getSnapshotUint8(int width, int height) const override;
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
