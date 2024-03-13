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
 *
 */

class RenderPipeline;
class GLViewer;
class ApplicationConfig;

/**
 * @brief Renderer class definition
 *
 */
class Renderer : public QObject {
  Q_OBJECT
 public:
  enum EVENT_TYPE : signed {
    ON_LEFT_CLICK = 0,
    ON_RIGHT_CLICK = 1,
    ON_MOUSEWHEEL_CLICK = 2,
    ON_MOUSEWHEEL_SCROLL_UP = 3,
    ON_MOUSEWHEEL_SCROLL_DOWN = 4,
  };

 private:
  Renderer();

 public:
  Renderer(unsigned width, unsigned height, GLViewer *widget = nullptr);

  virtual ~Renderer();

  void initialize(ApplicationConfig *app_conf);

  /**
   * @brief Method setting up the meshes , and the scene camera
   *
   * @return true If operation succeeded
   */
  bool prep_draw();

  /**
   * @brief Initiates the draw calls for all objects in the scene
   *
   */
  void draw();

  /**
   * @brief Cleans up the former scene and replaces it with a new
   *
   * @param new_scene New scene to be rendererd
   */
  void set_new_scene(std::pair<std::vector<Mesh *>, SceneTree> &new_scene);

  /**
   * @brief Checks if all Drawable objects in the scene have been initialized
   *
   * @return true If the scene is initialized and ready
   */
  bool scene_ready();

  /**
   * @brief Get the pointer to the MouseState structure
   *
   * @return MouseState* Pointer on the mouse_state variable , informations about the mouse positions , button clicked
   * etc
   * @see MouseState
   */
  MouseState *getMouseStatePointer() { return &mouse_state; };
  const MouseState *getConstMouseStatePointer() const { return &mouse_state; }

  /**
   * @brief Left click event behavior
   *
   */
  void onLeftClick();

  /**
   * @brief Right click event behavior
   *
   */
  void onRightClick() const;

  /**
   * @brief Left click release event behavior
   *
   */
  void onLeftClickRelease();

  /**
   * @brief Right click release event behavior
   *
   */
  void onRightClickRelease() const;

  /**
   * @brief Scroll down event behavior
   *
   */
  void onScrollDown() const;

  /**
   * @brief Scroll up event behavior
   *
   */
  void onScrollUp() const;

  /**
   * @brief Set the new screen dimensions
   *
   * @param width Width of the screen
   * @param height Height of the screen
   */
  void onResize(unsigned int width, unsigned int height);

  /**
   * @brief Set the Default Framebuffer ID
   *
   * @param id ID of the default framebuffer
   */
  void setDefaultFrameBufferId(unsigned id) { default_framebuffer_id = id; }

  /**
   * @brief Returns a pointer on the default framebuffer property
   *
   * @return unsigned*
   */
  unsigned int *getDefaultFrameBufferIdPointer() { return &default_framebuffer_id; }

  /**
   * @brief Set the Gamma Value object
   *
   * @param gamma
   */
  void setGammaValue(float gamma) const;

  /**
   * @brief Set the Exposure Value object
   *
   * @param exposure
   */
  void setExposureValue(float exposure) const;

  /**
   * @brief Set post process to default
   *
   */
  void setNoPostProcess() const;

  /**
   *	@brief  Set post process to edge
   */
  void setPostProcessEdge() const;

  /**
   *	@brief  Set post process to sharpen
   */
  void setPostProcessSharpen() const;

  /**
   * @brief Set post process to blurr
   */
  void setPostProcessBlurr() const;

  /**
   * @brief Resets the scene camera to default position
   *
   */
  void resetSceneCamera() const;

  /**
   * @brief Display meshes in point clouds
   *
   */
  void setRasterizerPoint();

  /**
   * @brief Standard rasterizer display in polygons
   *
   */
  void setRasterizerFill();

  /**
   * @brief Display meshes in wireframe
   *
   */
  void setRasterizerWireframe();

  /**
   * @brief Control if we display every mesh's bounding box
   *
   */
  void displayBoundingBoxes(bool display);

  /**
   * @brief Retrieves the current final scene reference
   *
   * @return const Scene&
   */
  const Scene &getConstScene() const { return *scene; }

  Scene &getScene() { return *scene; }

  RenderPipeline &getRenderPipeline() const { return *render_pipeline; }

  template<EVENT_TYPE type>
  void pushEvent(RENDERER_CALLBACK_ENUM callback_enum, std::any data) {
    event_callback_stack[type].push(std::pair<RENDERER_CALLBACK_ENUM, std::any>(callback_enum, data));
  }

  bool eventQueueEmpty(EVENT_TYPE type) const { return event_callback_stack[type].empty(); }

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
  MouseState mouse_state{};            /**<Pointer on the MouseState structure*/
  Dim2 screen_size{};                  /**<Dimensions of the renderer windows*/
  unsigned int default_framebuffer_id; /**<In the case the GUI uses other contexts and other framebuffers , we use this
                                          variable to reset the rendering to the default framebuffer*/
  LightingDatabase light_database;
  GLViewer *gl_widget{};
  std::unique_ptr<EnvmapTextureManager> envmap_manager;

 private:
  ApplicationConfig *global_config;
  static const int event_num_size = ON_MOUSEWHEEL_SCROLL_DOWN + 1;
  std::queue<std::pair<RENDERER_CALLBACK_ENUM, std::any>> event_callback_stack[event_num_size];
};

#endif
