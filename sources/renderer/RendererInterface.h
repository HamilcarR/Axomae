#ifndef RendererInterface_H
#define RendererInterface_H
#include "EventInterface.h"
#include "Image.h"
#include "RendererCallbacks.h"
class Scene;
class SceneTree;
class RenderPipeline;
class ApplicationConfig;
class Camera;
class Mesh;
class GLViewer;
class EnvmapTextureManager;
namespace controller::event {
  class Event;
}  // namespace controller::event

struct SceneChangeData {
  SceneTree *scene;
  std::vector<Mesh *> mesh_list;
};

class RendererInterface : public EventInterface {
 public:
  ~RendererInterface() override = default;
  virtual void initialize(ApplicationConfig *app_conf) = 0;
  /**
   * @brief Method setting up the meshes , and the scene camera
   */
  virtual bool prep_draw() = 0;
  virtual void draw() = 0;
  virtual void onResize(unsigned int width, unsigned int height) = 0;
  virtual void onHideEvent() = 0;
  virtual void setDefaultFrameBufferId(unsigned id) = 0;
  virtual void onClose() = 0;
  /**
   * @brief Cleans up the former scene and replaces it with a new one.
   */
  [[nodiscard]] virtual unsigned int *getDefaultFrameBufferIdPointer() = 0;
  virtual void setViewerWidget(GLViewer *widget) = 0;
  virtual void getScreenPixelColor(int x, int y, float r_screen_pixel_color[4]) = 0;
  virtual void prepSceneChange() = 0;
  virtual void setNewScene(const SceneChangeData &new_scene) = 0;
  [[nodiscard]] virtual RenderPipeline &getRenderPipeline() const = 0;
  [[nodiscard]] virtual Scene &getScene() const = 0;
  [[nodiscard]] virtual const EnvmapTextureManager &getCurrentEnvmapId() const = 0;
  [[nodiscard]] virtual image::ImageHolder<float> getSnapshotFloat(int width, int height) const = 0;
  [[nodiscard]] virtual image::ImageHolder<uint8_t> getSnapshotUint8(int width, int height) const = 0;
  [[nodiscard]] virtual const Camera *getCamera() const = 0;
  [[nodiscard]] virtual Camera *getCamera() = 0;
};

class IRenderer : public RendererInterface {
 public:
  ~IRenderer() override = default;
  virtual void setGammaValue(float gamma) = 0;
  virtual void setExposureValue(float exposure) = 0;
  virtual void setNoPostProcess() = 0;
  virtual void setPostProcessEdge() = 0;
  virtual void setPostProcessSharpen() = 0;
  virtual void setPostProcessBlurr() = 0;
  virtual void resetSceneCamera() = 0;
  virtual void setRasterizerPoint() = 0;
  virtual void setRasterizerFill() = 0;
  virtual void setRasterizerWireframe() = 0;
  virtual void displayBoundingBoxes(bool display) = 0;

  template<RENDERER_CALLBACK_ENUM function_flag, class... Args>
  void executeMethod(Args &&...args) {
    if constexpr (function_flag == SET_GAMMA)
      setGammaValue(std::forward<Args>(args)...);
    else if constexpr (function_flag == SET_EXPOSURE)
      setExposureValue(std::forward<Args>(args)...);
    else if constexpr (function_flag == SET_POSTPROCESS_NOPROCESS)
      setNoPostProcess(std::forward<Args>(args)...);
    else if constexpr (function_flag == SET_POSTPROCESS_SHARPEN)
      setPostProcessSharpen(std::forward<Args>(args)...);
    else if constexpr (function_flag == SET_POSTPROCESS_BLURR)
      setPostProcessBlurr(std::forward<Args>(args)...);
    else if constexpr (function_flag == SET_POSTPROCESS_EDGE)
      setPostProcessEdge(std::forward<Args>(args)...);
    else if constexpr (function_flag == SET_RASTERIZER_POINT)
      setRasterizerPoint(std::forward<Args>(args)...);
    else if constexpr (function_flag == SET_RASTERIZER_FILL)
      setRasterizerFill(std::forward<Args>(args)...);
    else if constexpr (function_flag == SET_RASTERIZER_WIREFRAME)
      setRasterizerWireframe(std::forward<Args>(args)...);
    else if constexpr (function_flag == SET_DISPLAY_BOUNDINGBOX)
      displayBoundingBoxes(std::forward<Args>(args)...);
    else if constexpr (function_flag == SET_DISPLAY_RESET_CAMERA)
      resetSceneCamera(std::forward<Args>(args)...);
    else {
      AX_UNREACHABLE
    }
  }
};
#endif  // RendererInterface_H
