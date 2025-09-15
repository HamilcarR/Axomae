#include "api_common.h"
#include "api_renderbuffer.h"
#include "private_includes.h"
#include <atomic>

namespace nova {

  struct RenderBufferPack {
    std::vector<float> front;
    std::vector<float> back;

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
          front[i * RenderBufferPack::CHANNELS_COLOR + c] = 0.f;
          back[i * RenderBufferPack::CHANNELS_COLOR + c] = 0.f;
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
      front.clear();
      back.clear();
    }
  };

  class NvRenderBuffer : public RenderBuffer {
    std::unique_ptr<RenderBufferPack> render_buffers;
    std::atomic<float *> swap_sync{};
    float *back_ptr{nullptr};
    float *front_ptr{nullptr};

   public:
    unsigned getChannel(texture::CHANNEL_TYPE type) const override {
      switch (type) {
        case texture::COLOR:
          return RenderBufferPack::CHANNELS_COLOR;
        case texture::DEPTH:
          return RenderBufferPack::CHANNELS_DEPTH;
        case texture::NORMAL:
          return RenderBufferPack::CHANNELS_NORMALS;
      }
      return texture::COLOR;
    }

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
        render_buffers->front.resize(width * height * RenderBufferPack::CHANNELS_COLOR);
        render_buffers->back.resize(width * height * RenderBufferPack::CHANNELS_COLOR);
        render_buffers->partial_buffer.resize(width * height * RenderBufferPack::CHANNELS_COLOR);
        render_buffers->accumulator_buffer.resize(width * height * RenderBufferPack::CHANNELS_COLOR);
        render_buffers->depth_buffer.resize(width * height * RenderBufferPack::CHANNELS_DEPTH);
        render_buffers->normal_buffer.resize(width * height * RenderBufferPack::CHANNELS_NORMALS);
      } catch (...) {
        return OUT_OF_MEMORY;
      }

      back_ptr = render_buffers->back.data();
      front_ptr = render_buffers->front.data();
      swap_sync.store(back_ptr, std::memory_order_relaxed);
      return SUCCESS;
    }

    FloatView getAccumulator() const override { return render_buffers->accumulator_buffer; }

    HdrBufferStruct getRenderBuffers() const override {
      HdrBufferStruct buffers;
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

    void swapBackBuffer() override {
      float *current_ptr = swap_sync.load(std::memory_order_relaxed);
      if (current_ptr == render_buffers->front.data()) {
        swap_sync.store(back_ptr, std::memory_order_relaxed);
      } else {
        swap_sync.store(front_ptr, std::memory_order_relaxed);
      }
      std::swap(back_ptr, front_ptr);
    }

    FloatView backBuffer() override { return FloatView(back_ptr, render_buffers->front.size()); }
    FloatView frontBuffer() override { return FloatView(front_ptr, render_buffers->front.size()); }

    RenderOutput getFrameBuffer() const override {
      RenderOutput output;
      output.color_buffer = FloatView(front_ptr, render_buffers->front.size());
      output.color_buffer_channels = render_buffers->CHANNELS_COLOR;
      output.color_buffer_width = render_buffers->width;
      output.color_buffer_height = render_buffers->height;

      output.depth_buffer = render_buffers->depth_buffer;
      output.depth_buffer_width = render_buffers->width;
      output.depth_buffer_height = render_buffers->height;

      output.normal_buffer = render_buffers->normal_buffer;
      output.normal_buffer_channels = render_buffers->CHANNELS_NORMALS;
      output.normal_buffer_width = render_buffers->width;
      output.normal_buffer_height = render_buffers->height;

      return output;
    }

    unsigned getWidth() const override { return render_buffers->width; }

    unsigned getHeight() const override { return render_buffers->height; }
  };

  RenderBufferPtr create_renderbuffer() { return std::make_unique<NvRenderBuffer>(); }
}  // namespace nova
