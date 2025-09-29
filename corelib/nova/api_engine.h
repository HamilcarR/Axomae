#ifndef API_ENGINE_H
#define API_ENGINE_H
#include "api_camera.h"
#include "api_common.h"
#include "api_geometry.h"
#include "api_material.h"
#include "api_renderbuffer.h"
#include "api_renderoptions.h"
#include "api_scene.h"
#include "api_texture.h"
#include "api_transform.h"

#include "integrator/integrator_includes.h"
#include "manager/manager_includes.h"
#include "material/material_includes.h"
#include "primitive/primitive_includes.h"
#include "shape/shape_includes.h"
#include "texturing/texturing_includes.h"
#include "utils/utils_includes.h"
#include <internal/geometry/Object3D.h>

namespace nova {

  /**
   * @brief Interface for engine-internal callback handlers.
   *
   * This abstract class allows users to define custom routines that are executed
   * before and after each render frame. Implementations can be used to manage
   * device contexts or perform per-frame setup/teardown.
   */
  class EngineUserCallbackHandler {
   public:
    virtual ~EngineUserCallbackHandler() = default;

    /**
     * @brief Called before each render frame.
     * Implement this to perform any setup required before rendering a frame,
     * such as setting the current GPU context.
     */
    virtual void framePreRenderRoutine() = 0;

    /**
     * @brief Called after each render frame.
     * Implement this to perform any cleanup or state restoration after rendering a frame,
     * such as resetting the GPU context.
     */
    virtual void framePostRenderRoutine() = 0;
  };

  /**
   * @brief Abstract interface for the rendering engine.
   *
   * This class defines the core API for interacting with a render engine,
   * including scene setup, acceleration structure building, render buffer management,
   * and resource cleanup.
   */
  class Engine {
   public:
    virtual ~Engine() = default;
    /**
     * @brief Builds the internal registred scene.
     *
     * @return ERROR_STATE
     *   - SUCCESS: Scene built successfully.
     *   - INVALID_SCENE_TYPE: The provided scene is not of a supported type.
     *   - OUT_OF_MEMORY: Failed to allocate memory for scene resources.
     */
    virtual ERROR_STATE buildScene() = 0;
    virtual ERROR_STATE setScene(ScenePtr scene) = 0;
    virtual const Scene *getScene() const = 0;
    virtual Scene *getScene() = 0;

    /**
     * @brief Builds the acceleration structure (e.g., BVH) for the current scene.
     *
     * buildScene() must be called successfully before this.
     * GPU acceleration structures are built if render options enable them.
     * @return ERROR_STATE
     *   - SUCCESS: Acceleration structure built successfully.
     *   - SCENE_NOT_PROCESSED: Scene has not been built yet.
     */
    virtual ERROR_STATE buildAcceleration() = 0;

    /**
     * @brief Moves a render buffer instance to the engine.
     *
     * @param render_buffer The render buffer to set.
     * @return ERROR_STATE
     *   - SUCCESS: Render buffer set successfully.
     *   - INVALID_BUFFER_STATE: Buffer is invalid or not compatible.
     */
    virtual ERROR_STATE setRenderBuffers(RenderBufferPtr render_buffer) = 0;

    virtual const RenderBuffer *getRenderBuffers() const = 0;

    virtual RenderBuffer *getRenderBuffers() = 0;

    /**
     * @brief Cleans up all resources associated with the engine.
     *
     * @return ERROR_STATE
     *   - SUCCESS: Cleanup successful.
     *   - INVALID_BUFFER_STATE: Render buffer is invalid/not allocated, and cannot be reset.
     */
    virtual ERROR_STATE cleanup() = 0;

    /**
     * @brief Sets the rendering options for the engine.
     * Configures the rendering behavior using the provided RenderOptions instance.
     * The options include settings such as sampling, ray depth, integrator flags, and tile dimensions.
     * @param opts A RenderOptionsPtr instance containing the desired rendering parameters.
     * @return ERROR_STATE
     *   - SUCCESS: Rendering options set successfully.
     *   - INVALID_ARGUMENT: opts is invalid.
     */
    virtual ERROR_STATE setRenderOptions(RenderOptionsPtr opts) = 0;
    /**
     * @brief Returns an allocated pointer to an image of the engine's options
     *
     * @return RenderOptions*
     */
    virtual const RenderOptions *getRenderOptions() const = 0;

    virtual RenderOptions *getRenderOptions() = 0;

    /**
     * @brief Stops an ongoing rendering process.
     * Halts the current render operation, releasing any active resources.
     * This method executes a complete synchronization and is fairly slow.
     * @return ERROR_STATE
     *   - SUCCESS: Rendering stopped successfully.
     *   - THREADPOOL_NOT_INITIALIZED: Invalid internal threadpool configuration.
     */
    virtual ERROR_STATE stopRender() = 0;

    /**
     * @brief Initiates a new rendering process.
     * This method starts the rendering pipeline with the current scene and render options.
     * It triggers the engine to begin processing the scene and writing output to the internal render buffers.
     *  @return ERROR_STATE
     *   - SUCCESS: Rendering started successfully.
     *   - INVALID_ARGUMENT: ctx_manager is null but renderer is set to use interops for GPGPU rendering.
     *   - SCENE_NOT_PROCESSED: Scene has not been built or is invalid.
     */
    virtual ERROR_STATE startRender() = 0;

    /**
     * @brief Sets a custom callback that executes before and after a drawcall.
     * @param user_callback Pointer to a callback functor, provides for the user a way to execute code inside the render loop before and after a
     * drawcall. Can be used for setting up context switching for cuda gl interops. Can be null.
     * @see EngineInternalCallbackHandler.
     *
     */
    virtual void setUserCallback(EngineUserCallbackHandlerPtr user_callback) = 0;

    /**
     * @brief Checks if the engine is currently rendering.
     *
     * This method returns true if a rendering process is currently active,
     * and false otherwise. It can be used to query the engine's state to
     * determine if rendering is in progress.
     * @return true if the engine is rendering, false otherwise.
     */
    virtual bool isRendering() const = 0;

    /**
     * @brief Prepares the engine for rendering by performing any necessary setup steps.
     * This may include initializing internal state, checking dependencies, or allocating temporary buffers.
     * It is typically called before startRender() to ensure a clean and ready state.
     * @return ERROR_STATE
     *   - SUCCESS: Preparation completed successfully.
     */
    virtual ERROR_STATE prepareRender() = 0;

    /**
     * @brief Retrieves the latest error message from the render engine.
     *
     * Fills the provided character array with a human-readable error message
     * describing the most recent error encountered by the render engine. The actual number
     * of characters written (excluding the null terminator) is returned via the 'size' parameter.
     *
     * @param error_log A character array of size 1024 to receive the error message.
     * @param size Reference to a size_t variable that will be set to the length of the error message.
     */
    virtual void getRenderEngineError(char error_log[1024], size_t &size) = 0;

    virtual unsigned getFrameIndex() const = 0;

    // TODO: DELETE AT THE END
    virtual NovaResourceManager &getResrcManager() = 0;
  };
  // TODO: Same with this.
  inline uint64_t get_error_status(const NovaExceptionManager &exception_manager) { return exception_manager.checkErrorStatus(); }
  inline std::vector<exception::ERROR> get_error_list(const nova::NovaExceptionManager &exception_manager) {
    return exception_manager.getErrorList();
  }

  EnginePtr create_engine();

  /**
   * @brief Sets the number of rendering threads to use.
   * This allows users to control the parallelism of the rendering process.
   * Slow function, introduces synchronization and thread pool re-creation.
   * @param threads Number of threads to use.
   * @return ERROR_STATE
   *   - SUCCESS: Thread count set successfully.
   *   - THREADPOOL_CREATION_ERROR: Failed creating threadpool.
   */

  ERROR_STATE init_threads(unsigned number_threads);

}  // namespace nova

#endif
