#ifndef NOVAAPI_H
#define NOVAAPI_H
#include "api_camera.h"
#include "api_common.h"
#include "api_geometry.h"
#include "api_material.h"
#include "api_scene.h"
#include "api_texture.h"
#include "engine/datastructures.h"
#include "integrator/integrator_includes.h"
#include "manager/manager_includes.h"
#include "material/material_includes.h"
#include "primitive/primitive_includes.h"
#include "shape/shape_includes.h"
#include "texturing/texturing_includes.h"
#include "utils/utils_includes.h"
#include <internal/geometry/Object3D.h>

namespace nova {

  class RenderBuffer;
  class Engine;
  class RenderOptions;
  using RenderBufferPtr = std::unique_ptr<RenderBuffer>;
  using EngineInstancePtr = std::unique_ptr<Engine>;
  using RenderOptionsPtr = std::unique_ptr<RenderOptions>;
  /**
   * @brief Encapsulates all the configurable parameters for rendering scenes in the Nova engine.
   * Allows users to fine-tune rendering behavior.
   */
  class RenderOptions {

   public:
    virtual ~RenderOptions() = default;

    /**
     * Sets the number of samples used for aliasing reduction.
     * @param s Number of aliasing samples.
     */
    virtual void setAliasingSamples(unsigned s) = 0;

    /**
     * Gets the current number of aliasing samples.
     * @return Number of aliasing samples.
     */
    virtual unsigned getAliasingSamples() const = 0;

    /**
     * Sets the maximum depth for ray tracing.
     * @param depth Maximum depth value.
     */
    virtual void setMaxDepth(unsigned depth) = 0;

    /**
     * Gets the current maximum ray tracing depth.
     * @return Maximum depth value.
     */
    virtual unsigned getMaxDepth() const = 0;

    /**
     * Sets the maximum number of samples per pixel.
     * @param samples Maximum sample count.
     */
    virtual void setMaxSamples(unsigned samples) = 0;

    /**
     * Gets the current maximum sample count per pixel.
     * @return Maximum sample count.
     */
    virtual unsigned getMaxSamples() const = 0;

    /**
     &nbsp;Sets the increment for sample count during progressive rendering.
     * @param inc Sample increment value.
     */
    virtual void setSamplesIncrement(unsigned inc) = 0;

    /**
     * Gets the current sample increment value.
     * @return Sample increment.
     */
    virtual unsigned getSamplesIncrement() const = 0;

    /**
     * Sets the dimensions of rendering tiles (width and height).
     * @param width Tile width.
     * @param height Tile height.
     */
    virtual void setTileDimension(unsigned width, unsigned height) = 0;

    /**
     * Gets the width of the rendering tile.
     * @return Tile width.
     */
    virtual unsigned getTileDimensionWidth() const = 0;

    /**
     * Gets the height of the rendering tile.
     * @return Tile height.
     */
    virtual unsigned getTileDimensionHeight() const = 0;

    virtual ERROR_STATE setIntegratorFlag(int flag) = 0;

    virtual int getIntegratorFlag() const = 0;

    /**
     * Flips the vertical axis of the output image (useful for certain display setups).
     */
    virtual void flipV() = 0;

    virtual bool isFlippedV() const = 0;
  };

  /**
   * @brief Abstract interface for a render buffer.
   * This class defines the API for creating and accessing render buffers that the engine will write to.
   */
  class RenderBuffer {
   public:
    virtual ~RenderBuffer() = default;

    /**
     * @brief Creates a render buffer of the given dimensions.
     * Data format is always RGBA32F.
     * @param width The width of the buffer.
     * @param height The height of the buffer.
     * @return ERROR_STATE
     *   - SUCCESS: Buffer created successfully.
     *   - OUT_OF_MEMORY: Failed to allocate memory for the buffer.
     *   - INVALID_BUFFER_STATE: Buffer could not be created due to invalid state.
     */
    virtual ERROR_STATE createRenderBuffer(unsigned width, unsigned height) = 0;

    /**
     * @brief Retrieves the underlying render buffer data.
     * @return HdrBufferStruct containing the buffer data.
     *   - If the buffer is not valid, the returned struct will be empty.
     */
    virtual HdrBufferStruct getRenderBuffers() const = 0;
  };

  /**
   * @brief Abstract interface for a rendering engine instance.
   *
   * This class defines the core API for interacting with a rendering engine,
   * including scene setup, acceleration structure building, render buffer management,
   * and resource cleanup.
   */
  class Engine {
   public:
    virtual ~Engine() = default;
    /**
     * @brief Builds the scene from the given abstract scene.
     *
     * @param scene The scene to build.
     * @return ERROR_STATE
     *   - SUCCESS: Scene built successfully.
     *   - INVALID_SCENE_TYPE: The provided scene is not of a supported type.
     *   - OUT_OF_MEMORY: Failed to allocate memory for scene resources.
     */
    virtual ERROR_STATE buildScene(const Scene &scene) = 0;

    /**
     * @brief Builds the acceleration structure (e.g., BVH) for the current scene.
     *
     * buildScene() must be called successfully before this.
     * GPU acceleration structures are built if useGpu() enables them.
     * @return ERROR_STATE
     *   - SUCCESS: Acceleration structure built successfully.
     *   - SCENE_NOT_PROCESSED: Scene has not been built yet.
     */
    virtual ERROR_STATE buildAcceleration() = 0;

    /**
     * @brief Enables or disables interop usage (e.g., CUDA/OpenGL interop).
     *
     * @param use True to enable interops, false to disable.
     * @return ERROR_STATE
     *   - SUCCESS: Interop state set.
     *   - NOT_GPU_BUILD: Interops not available in non-GPU builds.
     */
    virtual ERROR_STATE useInterops(bool use) = 0;

    /**
     * @brief Moves a render buffer instance to the engine.
     *
     * @param render_buffer The render buffer to set.
     * @return ERROR_STATE
     *   - SUCCESS: Render buffer set successfully.
     *   - INVALID_BUFFER_STATE: Buffer is invalid or not compatible.
     */
    virtual ERROR_STATE setRenderBuffers(RenderBufferPtr render_buffer) = 0;

    /**
     * @brief Returns an allocated pointer to an image of the engine's options
     *
     * @return RenderOptionsPtr
     */
    virtual const RenderOptions &getRenderOptions() const = 0;

    /**
     * @brief Enables or disables GPU usage for rendering.
     *
     * @param use True to use GPU, false to use CPU.
     * @return ERROR_STATE
     *   - SUCCESS: GPU usage state set.
     *   - NOT_GPU_BUILD: GPU usage not available in non-GPU builds.
     */
    virtual ERROR_STATE useGpu(bool use) = 0;

    /**
     * @brief Cleans up all resources associated with the engine.
     *
     * @return ERROR_STATE
     *   - SUCCESS: Cleanup successful.
     */
    virtual ERROR_STATE cleanup() = 0;

    virtual ERROR_STATE setRenderOptions(const RenderOptions &opts) = 0;

    virtual void stopRender() = 0;

    virtual void startRender() = 0;

    virtual ERROR_STATE synchronizeHost() = 0;

    virtual ERROR_STATE synchronizeDevice() = 0;

    virtual ERROR_STATE setThreadSize(unsigned threads) = 0;
  };

  RenderBufferPtr create_render_buffer();
  EngineInstancePtr create_engine();
  RenderOptionsPtr create_render_options();
  void errlog(int error, char log_buffer_r[128]);

}  // namespace nova

#endif
