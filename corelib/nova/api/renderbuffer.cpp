#include "private_includes.h"

namespace nova {

  struct RenderBufferPack {
    axstd::managed_vector<float> partial_buffer;
    axstd::managed_vector<float> accumulator_buffer;
    axstd::managed_vector<float> depth_buffer;
    axstd::managed_vector<float> normal_buffer;
    unsigned width{}, height{};
    static constexpr unsigned CHANNELS_COLOR = 4;
    static constexpr unsigned CHANNELS_DEPTH = 1;
    static constexpr unsigned CHANNELS_NORMALS = 3;

    void cleanCanvas() {
      for (size_t i = 0; i < width * height; i++) {
        for (size_t c = 0; c < RenderBufferPack::CHANNELS_COLOR; c++) {
          partial_buffer[i * RenderBufferPack::CHANNELS_COLOR + c] = 0.f;
          accumulator_buffer[i * RenderBufferPack::CHANNELS_COLOR + c] = 0.f;
        }
        for (size_t c = 0; c < RenderBufferPack::CHANNELS_DEPTH; c++)
          depth_buffer[i * RenderBufferPack::CHANNELS_DEPTH + c] = 0.f;
        for (size_t c = 0; c < RenderBufferPack::CHANNELS_NORMALS; c++)
          normal_buffer[i * RenderBufferPack::CHANNELS_NORMALS + c] = 0.f;
      }
    }

    void cleanup() {
      partial_buffer.clear();
      accumulator_buffer.clear();
      depth_buffer.clear();
      normal_buffer.clear();
    }
  };

  class NvRenderBuffer : public RenderBuffer {
    std::unique_ptr<RenderBufferPack> render_buffers;

   public:
    ERROR_STATE resetBuffers() override {
      if (!render_buffers)
        return INVALID_BUFFER_STATE;
      render_buffers->cleanCanvas();
      return SUCCESS;
    }

    ERROR_STATE createRenderBuffer(unsigned width, unsigned height) override {
      render_buffers = std::make_unique<RenderBufferPack>();
      render_buffers->width = width;
      render_buffers->height = height;
      try {
        render_buffers->partial_buffer.resize(width * height * RenderBufferPack::CHANNELS_COLOR);
        render_buffers->accumulator_buffer.resize(width * height * RenderBufferPack::CHANNELS_COLOR);
        render_buffers->depth_buffer.resize(width * height * RenderBufferPack::CHANNELS_DEPTH);
        render_buffers->normal_buffer.resize(width * height * RenderBufferPack::CHANNELS_NORMALS);
      } catch (...) {
        return OUT_OF_MEMORY;
      }
      return SUCCESS;
    }

    HdrBufferStruct getRenderBuffers() const override {
      HdrBufferStruct buffers;
      buffers.accumulator_buffer = render_buffers->accumulator_buffer.data();
      buffers.partial_buffer = render_buffers->partial_buffer.data();
      buffers.normal_buffer = render_buffers->normal_buffer.data();
      buffers.depth_buffer = render_buffers->depth_buffer.data();
      buffers.channels = render_buffers->CHANNELS_COLOR;
      buffers.byte_size_color_buffers = render_buffers->width * render_buffers->height * render_buffers->CHANNELS_COLOR * sizeof(float);
      buffers.color_buffers_pitch = render_buffers->width * render_buffers->CHANNELS_COLOR * sizeof(float);
      buffers.depth_buffers_pitch = render_buffers->width * render_buffers->CHANNELS_DEPTH * sizeof(float);
      buffers.normal_buffers_pitch = render_buffers->width * render_buffers->CHANNELS_NORMALS * sizeof(float);
      return buffers;
    }
  };

  RenderBufferPtr create_renderbuffer() { return std::make_unique<NvRenderBuffer>(); }
}  // namespace nova
