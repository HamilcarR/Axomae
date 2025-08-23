#ifndef API_RENDEROPTIONS_H
#define API_RENDEROPTIONS_H
#include "api_common.h"

namespace nova {
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

    /**
     * @brief Sets a flag in the integrator configuration. The flag must be one of the utility flags
     * (NORMAL, DEPTH, SPECULAR, DIFFUSE, EMISSIVE) or a combination of them, combined with one
     * of the first 7 flags (PATH, BIPATH, SPECTRAL, METROPOLIS, PHOTON, MARCHING, HYBRID)
     * For ex , PATH | NORMAL | DEPTH | SPECULAR is valid, whereas PATH | SPECTRAL | DEPTH is not.
     * @param flag A valid integrator flag from integrator::TYPE.
     * @return ERROR_STATE
     *    - SUCCESS: if the flag was set successfully.
     *    - MULTIPLE_INTEGRATORS_NOT_SUPPORTED: If the flag contains a bad combination of integrator types.
     */
    virtual ERROR_STATE setIntegratorFlag(int flag) = 0;

    /**
     * @brief Gets the currently set integrator flag value.
     * @return Integrator flag value.
     */
    virtual int getIntegratorFlag() const = 0;

    /**
     * @brief Enables or disables GPU usage for rendering.
     *
     * @param use True to use GPU, false to use CPU.
     * @return ERROR_STATE
     *   - SUCCESS: GPU usage state set.
     *   - NOT_GPU_BUILD: GPU usage not available in non-GPU builds.
     */
    virtual ERROR_STATE useGpu(bool use) = 0;

    virtual bool isUsingGpu() const = 0;

    /**
     * @brief Enables or disables interop usage (e.g., CUDA/OpenGL interop).
     *
     * @param use True to enable interops, false to disable.
     * @return ERROR_STATE
     *   - SUCCESS: Interop state set.
     *   - NOT_GPU_BUILD: Interops not available in non-GPU builds.
     */
    virtual ERROR_STATE useInterops(bool use) = 0;

    virtual bool isUsingInterops() const = 0;

    /**
     * Flips the vertical axis of the output image (useful for certain display setups).
     */
    virtual void flipV() = 0;

    virtual bool isFlippedV() const = 0;
  };

  RenderOptionsPtr create_renderoptions();

}  // namespace nova
#endif  // API_RENDEROPTIONS_H
