#ifndef API_RENDERBUFFER_H
#define API_RENDERBUFFER_H
#include "api_common.h"
#include "engine/datastructures.h"
namespace nova {
  struct Framebuffer {
    FloatView color_buffer{};
    unsigned color_buffer_channels{};
    unsigned color_buffer_width{};
    unsigned color_buffer_height{};

    FloatView depth_buffer{};
    unsigned depth_buffer_width{};
    unsigned depth_buffer_height{};

    FloatView normal_buffer{};
    unsigned normal_buffer_channels{};
    unsigned normal_buffer_width{};
    unsigned normal_buffer_height{};
  };

  struct frgba_s {
    float r, g, b, a;
  };

  struct frgb_s {
    float r, g, b;
  };

  /**
   * @brief Abstract interface for a render buffer.
   *
   * This class defines the API for creating and accessing render buffers that the Nova rendering
   * engine will write to. A render buffer manages multiple types of rendering data including:
   * - Color buffers (RGBA32F format) for final rendered images
   * - Depth buffers for depth testing and post-processing
   * - Normal buffers for lighting calculations and post-processing effects
   * - Accumulator buffers for progressive rendering techniques
   *
   * The buffer supports double buffering with front/back buffer swapping for smooth rendering.
   *
   * @note All buffer data is stored in 32-bit floating point format for high dynamic range support.
   * @note The buffer dimensions are uniform across all buffer types (color, depth, normal).
   */
  class RenderBuffer {
   public:
    virtual ~RenderBuffer() = default;

    /**
     * @brief Preallocate internal buffers to the maximum required size.
     *
     * This method allocates memory for all internal buffers (color, depth, normal, accumulator, etc.)
     * to accommodate the specified maximum width and height. This is useful for avoiding repeated
     * allocations when resizing the render buffer multiple times, as it allows for efficient reuse
     * of preallocated memory.
     *
     * @param width The maximum width to preallocate.
     * @param height The maximum height to preallocate.
     * @return ERROR_STATE
     *    - SUCCESS: Preallocation succeeded.
     *    - OUT_OF_MEMORY: Failed to allocate memory.
     */
    virtual ERROR_STATE preallocate(unsigned width, unsigned height) = 0;

    /**
     * @brief Set the active dimensions of the render buffer.
     *
     * This method sets the current width and height of the render buffer, and updates all internal
     * buffers. Must call preallocate() first.
     *
     * @param width The new width of the buffer.
     * @param height The new height of the buffer.
     * @return ERROR_STATE
     *   - SUCCESS: Buffer resized successfully.
     *   - OUT_OF_MEMORY: Failed to allocate memory for the buffer.
     */
    virtual ERROR_STATE resize(unsigned width, unsigned height) = 0;

    /**
     * @brief Resets all underlying buffers (color, depth, normal) to zero.
     * This operation writes 0 to all the underlying buffers, effectively clearing the render state.
     * Useful for initializing a render buffer before a new rendering pass.
     * @return ERROR_STATE
     *   - SUCCESS: All buffers have been successfully reset.
     *   - INVALID_BUFFER_STATE: Buffer is not in a valid state for reset (Not initialized).
     */
    virtual ERROR_STATE resetBuffers() = 0;

    /**
     * @brief Retrieves the underlying render buffer data.
     * @return HdrBufferStruct containing the buffer data.
     *   - If the buffer is not valid, the returned struct will be empty.
     */
    virtual HdrBufferStruct getRenderBuffers() const = 0;

    /**
     * @brief Gets the back buffer for rendering operations.
     * The back buffer is where the renderer writes new frame data while the front buffer
     * displays the previously rendered frame. This enables double buffering for smooth rendering.
     * @return FloatView A view into the back buffer data (RGBA32F format).
     * @note The returned view is valid only while the buffer is initialized.
     */
    virtual FloatView backBuffer() = 0;

    /**
     * @brief Gets the front buffer for display operations.
     * The front buffer contains the currently displayed frame data. This is typically
     * what gets presented to the user or copied to display textures.
     * @return FloatView A view into the front buffer data (RGBA32F format).
     * @note The returned view is valid only while the buffer is initialized.
     */
    virtual FloatView frontBuffer() = 0;

    /**
     * @brief Swaps the front and back buffers atomically.
     * This operation performs a double-buffer swap, making the current back buffer
     * become the new front buffer and vice versa. The operation is thread-safe and
     * uses atomic operations to ensure consistency across multiple threads.
     * @note This method should be called after completing a frame render to present
     *       the new frame and prepare for the next rendering pass.
     */
    virtual void swapBackBuffer() = 0;

    /**
     * @brief Gets the accumulator buffer for progressive rendering.
     * The accumulator buffer stores accumulated color values across multiple rendering
     * passes, enabling techniques like progressive rendering and denoising.
     * @return FloatView A view into the accumulator buffer data (RGBA32F format).
     * @note The accumulator buffer is automatically cleared when resetBuffers() is called.
     */
    virtual FloatView getAccumulator() const = 0;

    /**
     * @brief Gets the complete frame buffer output containing all render data.
     * This method returns a structured output containing color, depth, and normal
     * buffer data along with their dimensions and channel information.
     * @return Framebuffer A structure containing:
     *   - color_buffer: RGBA color data (4 channels)
     *   - depth_buffer: Depth buffer data (1 channel)
     *   - normal_buffer: Normal buffer data (3 channels)
     *   - Associated width, height, and channel count for each buffer
     * @note All buffers share the same width and height dimensions.
     */
    virtual Framebuffer getFramebuffer() const = 0;

    /**
     * @brief Gets the width of the render buffer.
     * @return unsigned The width in pixels of all buffers (color, depth, normal).
     * @note Returns 0 if the buffer has not been initialized.
     */
    virtual unsigned getWidth() const = 0;

    /**
     * @brief Gets the height of the render buffer.
     * @return unsigned The height in pixels of all buffers (color, depth, normal).
     * @note Returns 0 if the buffer has not been initialized.
     */
    virtual unsigned getHeight() const = 0;

    /**
     * @brief Gets the number of channels for a specific buffer type.
     * @param type The type of buffer to query (COLOR, DEPTH, or NORMAL).
     * @return unsigned The number of channels for the specified buffer type:
     *   - COLOR: 4 channels (RGBA)
     *   - DEPTH: 1 channel (depth value)
     *   - NORMAL: 3 channels (XYZ normal vector)
     * @note Returns COLOR channel count for unknown types.
     */
    virtual unsigned getChannel(texture::CHANNEL_TYPE type) const = 0;

    virtual frgba_s sampleColor(unsigned x, unsigned y) const = 0;
    virtual frgb_s sampleNormal(unsigned x, unsigned y) const = 0;
    virtual float sampleDepth(unsigned x, unsigned y) const = 0;
  };

  /**
   * @brief Factory function to create a new render buffer instance.
   * Creates a new instance of the default render buffer implementation.
   * The returned buffer is uninitialized and must be configured using createRenderBuffer().
   * @return RenderBufferPtr A unique pointer to a new RenderBuffer instance.
   * @note The caller is responsible for managing the lifetime of the returned buffer.
   */
  RenderBufferPtr create_renderbuffer();

}  // namespace nova
#endif
