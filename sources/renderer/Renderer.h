#ifndef RENDERER_H
#define RENDERER_H
#include "Camera.h"
#include "CameraFrameBuffer.h"
#include "EnvmapTextureManager.h"
#include "GLViewer.h"
#include "LightingDatabase.h"
#include "RendererInterface.h"
#include "ResourceDatabaseManager.h"
#include "Scene.h"
#include "constants.h"
/**
 * @file Renderer.h
 * Implementation of the renderer system
 */

class RenderPipeline;
class GLViewer;
class ApplicationConfig;
namespace controller {
  namespace event {
    class Event;
  }
}  // namespace controller
/**
 * @brief Renderer class definition
 */
class Renderer final : public IRenderer {
 public:
  std::unique_ptr<RenderPipeline> render_pipeline;
  std::unique_ptr<CameraFrameBuffer> camera_framebuffer;
  bool start_draw;
  ResourceDatabaseManager *resource_database;
  std::unique_ptr<Scene> scene;
  Camera *scene_camera;
  Dim2 screen_size{};
  unsigned int default_framebuffer_id;
  LightingDatabase light_database;
  GLViewer *gl_widget{};
  std::unique_ptr<EnvmapTextureManager> envmap_manager;

 private:
  Renderer();

 public:
  Renderer(unsigned width, unsigned height, GLViewer *widget = nullptr);
  Renderer(const Renderer &copy) = delete;
  Renderer(Renderer &&move) noexcept;
  Renderer &operator=(const Renderer &copy) = delete;
  Renderer &operator=(Renderer &&move) noexcept;
  ~Renderer() override;
  /**
   * @brief Method setting up the meshes , and the scene camera
   */
  bool prep_draw() override;
  ax_no_discard const EnvmapTextureManager &getCurrentEnvmapId() const override { return *envmap_manager; }
  void setNewScene(const SceneChangeData &new_scene) override;
  /**
   * @brief Checks if all Drawable objects in the scene have been initialized
   */
  bool scene_ready();
  void initialize(ApplicationConfig *app_conf) override;
  void draw() override;
  void processEvent(const controller::event::Event *event) override;
  void onResize(unsigned int width, unsigned int height) override;
  void onClose() override;
  void setDefaultFrameBufferId(unsigned id) override { default_framebuffer_id = id; }
  void getScreenPixelColor(int x, int y, float r_screen_pixel_color[4]) override;
  ax_no_discard unsigned int *getDefaultFrameBufferIdPointer() override { return &default_framebuffer_id; }
  ax_no_discard const Scene &getConstScene() const { return *scene; }
  ax_no_discard Scene &getScene() const override { return *scene; }
  ax_no_discard RenderPipeline &getRenderPipeline() const override { return *render_pipeline; }
  ax_no_discard image::ImageHolder<float> getSnapshotFloat(int width, int height) const override;
  ax_no_discard image::ImageHolder<uint8_t> getSnapshotUint8(int width, int height) const override;
  ax_no_discard const Camera *getCamera() const override { return scene_camera; }
  ax_no_discard Camera *getCamera() override { return scene_camera; }
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
  void prepSceneChange() override;
  void onHideEvent() override;
};

#endif
