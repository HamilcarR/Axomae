#ifndef API_TEXTURE_H
#define API_TEXTURE_H
#include "api_common.h"
#include <cstdint>
#include <internal/device/gpgpu/device_transfer_interface.h>

namespace nova {

  /**
   * @brief Abstract interface for texture management in the Nova rendering engine.
   * 
   * The Texture class provides a unified interface for managing texture data across
   * different rendering backends. It supports both CPU and GPU memory management,
   * OpenGL interop, and various texture formats.
   * 
   * Supported texture formats:
   * - UINT8X4: 4-channel unsigned 8-bit integer format (RGBA)
   * - FLOATX4: 4-channel 32-bit floating-point format (RGBA)
   * 
   * The texture supports 1-4 channels and can be inverted along X and Y axes.
   * OpenGL interop is supported for seamless integration with OpenGL rendering.
   */
  class Texture {
   public:
    virtual ~Texture() = default;

    /**
     * @brief Template method to set texture buffer data.
     * 
     * This is a convenience template that automatically calls the appropriate
     * setData() method based on the data type. Supported types are:
     * - const uint32_t*: Sets UINT8X4 format
     * - const float*: Sets FLOATX4 format
     * 
     * @tparam T Data type of the texture buffer
     * @param data Pointer to texture data buffer
     * @return ERROR_STATE SUCCESS on success, INVALID_BUFFER_STATE if data is null
     */
    template<class T>
    ERROR_STATE setTextureBuffer(const T *data) {
      return setData(data);
    }

    /**
     * @brief Set the width of the texture in pixels.
     * 
     * @param w Width in pixels (must be > 0)
     * @return ERROR_STATE Always returns SUCCESS
     */
    virtual ERROR_STATE setWidth(unsigned w) = 0;

    /**
     * @brief Set the height of the texture in pixels.
     * 
     * @param h Height in pixels (must be > 0)
     * @return ERROR_STATE Always returns SUCCESS
     */
    virtual ERROR_STATE setHeight(unsigned h) = 0;

    /**
     * @brief Set the number of channels in the texture.
     * 
     * @param c Number of channels (must be between 1 and 4 inclusive)
     * @return ERROR_STATE SUCCESS on success, INVALID_CHANNEL_DESCRIPTOR if c < 1 or c > 4
     */
    virtual ERROR_STATE setChannels(unsigned c) = 0;

    /**
     * @brief Set the OpenGL texture ID for interop operations.
     * 
     * This allows the texture to be used with OpenGL textures for seamless
     * integration between the Nova renderer and OpenGL rendering pipeline.
     * 
     * @param texture_id OpenGL texture ID
     * @return ERROR_STATE Always returns SUCCESS
     */
    virtual ERROR_STATE setInteropID(GLuint texture_id) = 0;

    /**
     * @brief Toggle Y-axis inversion of the texture.
     * 
     * This method toggles the current Y-axis inversion state. Useful for
     * correcting texture orientation when loading from different image formats
     * or coordinate systems.
     * 
     * @return ERROR_STATE Always returns SUCCESS
     */
    virtual ERROR_STATE invertY() = 0;

    /**
     * @brief Toggle X-axis inversion of the texture.
     * 
     * This method toggles the current X-axis inversion state. Useful for
     * correcting texture orientation when loading from different image formats
     * or coordinate systems.
     * 
     * @return ERROR_STATE Always returns SUCCESS
     */
    virtual ERROR_STATE invertX() = 0;

    /**
     * @brief Get a pointer to the texture buffer data.
     * 
     * The returned pointer type depends on the texture format:
     * - UINT8X4 format: returns const uint32_t*
     * - FLOATX4 format: returns const float*
     * 
     * @return const void* Pointer to texture data buffer
     */
    virtual const void *getTextureBuffer() const = 0;

    /**
     * @brief Get the width of the texture in pixels.
     * 
     * @return unsigned Width in pixels
     */
    virtual unsigned getWidth() const = 0;

    /**
     * @brief Get the height of the texture in pixels.
     * 
     * @return unsigned Height in pixels
     */
    virtual unsigned getHeight() const = 0;

    /**
     * @brief Get the number of channels in the texture.
     * 
     * @return unsigned Number of channels (1-4)
     */
    virtual unsigned getChannels() const = 0;

    /**
     * @brief Get the OpenGL texture ID for interop operations.
     * 
     * @return GLuint OpenGL texture ID (0 if not set)
     */
    virtual GLuint getInteropID() const = 0;

    /**
     * @brief Check if the texture is inverted along the Y-axis.
     * 
     * @return bool true if Y-axis is inverted, false otherwise
     */
    virtual bool getInvertY() const = 0;

    /**
     * @brief Check if the texture is inverted along the X-axis.
     * 
     * @return bool true if X-axis is inverted, false otherwise
     */
    virtual bool getInvertX() const = 0;

    /**
     * @brief Get the texture format.
     * 
     * @return texture::FORMAT The current texture format (UINT8X4 or FLOATX4)
     */
    virtual texture::FORMAT getFormat() const = 0;

   protected:
    /**
     * @brief Set texture data from a uint32_t buffer.
     * 
     * This method sets the texture data and automatically configures the format
     * to UINT8X4. The buffer is expected to contain 4-channel RGBA data.
     * 
     * @param buffer Pointer to uint32_t texture data
     * @return ERROR_STATE SUCCESS on success, INVALID_BUFFER_STATE if buffer is null
     */
    virtual ERROR_STATE setData(const uint32_t *buffer) = 0;

    /**
     * @brief Set texture data from a float buffer.
     * 
     * This method sets the texture data and automatically configures the format
     * to FLOATX4. The buffer is expected to contain 4-channel RGBA data.
     * 
     * @param buffer Pointer to float texture data
     * @return ERROR_STATE SUCCESS on success, INVALID_BUFFER_STATE if buffer is null
     */
    virtual ERROR_STATE setData(const float *buffer) = 0;
  };

  /**
   * @brief Factory function to create a new texture instance.
   * 
   * Creates a new texture using the default implementation.
   * The returned texture is uninitialized and requires setting dimensions,
   * channels, and data before use.
   * 
   * @return TexturePtr Unique pointer to a new Texture instance
   */
  TexturePtr create_texture();
}  // namespace nova

#endif
