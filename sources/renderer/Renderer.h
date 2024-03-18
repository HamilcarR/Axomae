#ifndef RENDERER_H
#define RENDERER_H

#include <any>
#include <queue>

#include "Camera.h"
#include "CameraFrameBuffer.h"
#include "Drawable.h"
#include "EnvmapTextureManager.h"
#include "GLViewer.h"
#include "LightingDatabase.h"
#include "Loader.h"
#include "Mesh.h"
#include "RendererEnums.h"
#include "ResourceDatabaseManager.h"
#include "Scene.h"
#include "constants.h"
#include "utils_3D.h"
/**
 * @file Renderer.h
 * Implementation of the renderer system
 */

class RenderPipeline;
class GLViewer;
class ApplicationConfig;
namespace controller::event {
  class Event;
}
/**
 * @brief Renderer class definition
 */
class Renderer : public QObject {
  Q_OBJECT

 private:
  Renderer();

 public:
  Renderer(unsigned width, unsigned height, GLViewer *widget = nullptr);

  ~Renderer() override;

  void initialize(ApplicationConfig *app_conf);

  /**
   * @brief Method setting up the meshes , and the scene camera
   */
  bool prep_draw();

  void draw();

  /**
   * @brief Cleans up the former scene and replaces it with a new
   * @param new_scene New scene to be rendererd
   */
  void set_new_scene(std::pair<std::vector<Mesh *>, SceneTree> &new_scene);

  /**
   * @brief Checks if all Drawable objects in the scene have been initialized
   * @return true If the scene is initialized and ready
   */
  bool scene_ready();

  void processEvent(const controller::event::Event *event) const;

  void onResize(unsigned int width, unsigned int height);

  void setDefaultFrameBufferId(unsigned id) { default_framebuffer_id = id; }

  unsigned int *getDefaultFrameBufferIdPointer() { return &default_framebuffer_id; }

  void setGammaValue(float gamma) const;

  void setExposureValue(float exposure) const;

  void setNoPostProcess() const;

  void setPostProcessEdge() const;

  void setPostProcessSharpen() const;

  void setPostProcessBlurr() const;

  void resetSceneCamera() const;

  void setRasterizerPoint() const;

  void setRasterizerFill() const;

  void setRasterizerWireframe() const;

  void displayBoundingBoxes(bool display) const;

  [[nodiscard]] const Scene &getConstScene() const { return *scene; }

  [[nodiscard]] Scene &getScene() const { return *scene; }

  [[nodiscard]] RenderPipeline &getRenderPipeline() const { return *render_pipeline; }

  /**
   * @brief Executes a specific method according to the value of the callback flag
   *
   * @tparam Args Type of the arguments
   * @param callback_flag	A flag of type RENDERER_CALLBACK_ENUM , will call one of the renderer methods
   * @param args Arguments of the callback function
   */
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
      return;
    }
  }

  template<class Func, class... Args>
  void execCallback(Func &&function, Renderer *instance, Args &&...args) {
    function(instance, std::forward<Args>(args)...);
  }

 signals:
  void sceneModified();
  void envmapSelected();

 public:
  std::unique_ptr<RenderPipeline> render_pipeline;
  std::unique_ptr<CameraFrameBuffer> camera_framebuffer; /**<Main framebuffer attached to the view*/
  bool start_draw;                                       /**<If the renderer is ready to draw*/
  ResourceDatabaseManager
      &resource_database;              /**<The main database containing a texture database ,a node database for stored meshes and a shader database*/
  std::unique_ptr<Scene> scene;        /**<The scene to be rendered*/
  Camera *scene_camera;                /**<Pointer on the scene camera*/
  Dim2 screen_size{};                  /**<Dimensions of the renderer windows*/
  unsigned int default_framebuffer_id; /**<In the case the GUI uses other contexts and other framebuffers , we use this
                                          variable to reset the rendering to the default framebuffer*/
  LightingDatabase light_database;
  GLViewer *gl_widget{};
  std::unique_ptr<EnvmapTextureManager> envmap_manager;
};

#endif
