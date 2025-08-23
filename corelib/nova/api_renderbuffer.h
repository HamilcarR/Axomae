#ifndef API_RENDERBUFFER_H
#define API_RENDERBUFFER_H
#include "api_common.h"
#include "engine/datastructures.h"
namespace nova {

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
  };

  RenderBufferPtr create_renderbuffer();

}  // namespace nova
#endif
