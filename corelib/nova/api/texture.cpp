#include "private_includes.h"
#include <memory>
namespace nova {

  class NvTexture final : public Texture {
   private:
    union {
      const float *f_buffer;
      const uint32_t *ui_buffer;
    } memory{};
    texture::FORMAT type{};
    unsigned width{0};
    unsigned height{0};
    unsigned channel{0};
    GLuint interop_id{0};
    bool invert_y{false};
    bool invert_x{false};

   public:
    NvTexture() = default;

    NvTexture(const Texture &other) {
      width = other.getWidth();
      height = other.getHeight();
      channel = other.getChannels();
      interop_id = other.getInteropID();
      invert_y = other.getInvertY();
      invert_x = other.getInvertX();
      type = other.getFormat();
      memory.ui_buffer = static_cast<const uint32_t *>(other.getTextureBuffer());
    }

    NvTexture &operator=(const Texture &other) {
      if (this == &other)
        return *this;
      width = other.getWidth();
      height = other.getHeight();
      channel = other.getChannels();
      interop_id = other.getInteropID();
      invert_y = other.getInvertY();
      invert_x = other.getInvertX();
      type = other.getFormat();
      memory.ui_buffer = static_cast<const uint32_t *>(other.getTextureBuffer());
      return *this;
    }

    ERROR_STATE setData(const uint32_t *buffer) override {
      if (!buffer)
        return INVALID_BUFFER_STATE;
      memory.ui_buffer = buffer;
      type = texture::UINT8X4;
      return SUCCESS;
    }

    ERROR_STATE setData(const float *buffer) override {
      if (!buffer)
        return INVALID_BUFFER_STATE;
      memory.f_buffer = buffer;
      type = texture::FLOATX4;
      return SUCCESS;
    }

    ERROR_STATE setWidth(unsigned w) override {
      width = w;
      return SUCCESS;
    }

    ERROR_STATE setHeight(unsigned h) override {
      height = h;
      return SUCCESS;
    }

    ERROR_STATE setChannels(unsigned c) override {
      if (c < 1 || c > 4)
        return INVALID_CHANNEL_DESCRIPTOR;
      channel = c;
      return SUCCESS;
    }

    ERROR_STATE setInteropID(GLuint texture_id) override {
      interop_id = texture_id;
      return SUCCESS;
    }

    ERROR_STATE invertY() override {
      invert_y = !invert_y;
      return SUCCESS;
    }

    ERROR_STATE invertX() override {
      invert_x = !invert_x;
      return SUCCESS;
    }

    const void *getTextureBuffer() const override { return memory.f_buffer; }

    texture::FORMAT getFormat() const override { return type; }

    unsigned getWidth() const override { return width; }

    unsigned getHeight() const override { return height; }

    unsigned getChannels() const override { return channel; }

    GLuint getInteropID() const override { return interop_id; }

    bool getInvertY() const override { return invert_y; }

    bool getInvertX() const override { return invert_x; }
  };

  TexturePtr create_texture() { return std::make_unique<NvTexture>(); }
}  // namespace nova
