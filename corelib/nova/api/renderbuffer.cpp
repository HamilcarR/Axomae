#include "api_common.h"
#include "api_renderbuffer.h"
#include "private_includes.h"
#include <atomic>
namespace nova {

  class RenderBufferPack {

    std::vector<float> front;
    std::vector<float> back;

    axstd::managed_vector<float> partial_buffer;
    axstd::managed_vector<float> accumulator_buffer;
    axstd::managed_vector<float> depth_buffer;
    axstd::managed_vector<float> normal_buffer;

    unsigned width{}, height{};
    size_t capacity{};
    bool is_prealloc{false};

    template<class T>
    FloatView toSpan(T &collection, unsigned channels) {
      return FloatView(collection.data(), width * height * channels);
    }

   public:
    static constexpr unsigned CHANNELS_COLOR = 4;
    static constexpr unsigned CHANNELS_DEPTH = 1;
    static constexpr unsigned CHANNELS_NORMALS = 3;

    void allocate(unsigned max_width, unsigned max_height) {
      is_prealloc = true;
      capacity = max_width * max_height;
      front.resize(capacity * CHANNELS_COLOR);
      back.resize(capacity * CHANNELS_COLOR);
      partial_buffer.resize(capacity * CHANNELS_COLOR);
      accumulator_buffer.resize(capacity * CHANNELS_COLOR);
      depth_buffer.resize(capacity * CHANNELS_DEPTH);
      normal_buffer.resize(capacity * CHANNELS_NORMALS);
    }

    void cleanCanvas() {
      for (size_t i = 0; i < width * height; i++) {
        for (size_t c = 0; c < CHANNELS_COLOR; c++) {
          partial_buffer[i * CHANNELS_COLOR + c] = 0.f;
          accumulator_buffer[i * CHANNELS_COLOR + c] = 0.f;
          front[i * CHANNELS_COLOR + c] = 0.f;
          back[i * CHANNELS_COLOR + c] = 0.f;
        }
        for (size_t c = 0; c < CHANNELS_DEPTH; c++)
          depth_buffer[i * CHANNELS_DEPTH + c] = 0.f;
        for (size_t c = 0; c < CHANNELS_NORMALS; c++)
          normal_buffer[i * CHANNELS_NORMALS + c] = 0.f;
      }
    }
    void setDimensions(unsigned w, unsigned h) {
      width = w;
      height = h;
    }

    size_t getCapacity() { return capacity; }

    unsigned getWidth() { return width; }

    unsigned getHeight() { return height; }

    FloatView frontBuffer() { return toSpan(front, CHANNELS_COLOR); }

    FloatView backBuffer() { return toSpan(back, CHANNELS_COLOR); }

    FloatView partialBuffer() { return toSpan(partial_buffer, CHANNELS_COLOR); }

    FloatView accumulatorBuffer() { return toSpan(accumulator_buffer, CHANNELS_COLOR); }

    FloatView depthBuffer() { return toSpan(depth_buffer, CHANNELS_DEPTH); }

    FloatView normalBuffer() { return toSpan(normal_buffer, CHANNELS_NORMALS); }
  };

  class NvRenderBuffer : public RenderBuffer {
    std::unique_ptr<RenderBufferPack> render_buffers;
    std::atomic<float *> swap_sync{};
    float *back_ptr{nullptr};
    float *front_ptr{nullptr};

   public:
    NvRenderBuffer() { render_buffers = std::make_unique<RenderBufferPack>(); }

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

    ERROR_STATE preallocate(unsigned width, unsigned height) override {
      try {
        render_buffers->allocate(width, height);
      } catch (...) {
        return OUT_OF_MEMORY;
      }
      back_ptr = render_buffers->backBuffer().data();
      front_ptr = render_buffers->frontBuffer().data();
      swap_sync.store(back_ptr, std::memory_order_relaxed);
      return SUCCESS;
    }

    ERROR_STATE resize(unsigned width, unsigned height) override {
      bool need_realloc = render_buffers->getCapacity() < width * height;
      ERROR_STATE error = SUCCESS;
      if (need_realloc)
        error = preallocate(width, height);
      render_buffers->setDimensions(width, height);

      return error;
    }

    FloatView getAccumulator() const override { return render_buffers->accumulatorBuffer(); }

    HdrBufferStruct getRenderBuffers() const override {
      HdrBufferStruct buffers;
      buffers.partial_buffer = render_buffers->partialBuffer().data();
      buffers.normal_buffer = render_buffers->normalBuffer().data();
      buffers.depth_buffer = render_buffers->depthBuffer().data();
      buffers.channels = static_cast<int>(RenderBufferPack::CHANNELS_COLOR);
      buffers.byte_size_color_buffers = render_buffers->getWidth() * render_buffers->getHeight() * render_buffers->CHANNELS_COLOR * sizeof(float);
      buffers.color_buffers_pitch = render_buffers->getWidth() * render_buffers->CHANNELS_COLOR * sizeof(float);
      buffers.depth_buffers_pitch = render_buffers->getWidth() * render_buffers->CHANNELS_DEPTH * sizeof(float);
      buffers.normal_buffers_pitch = render_buffers->getWidth() * render_buffers->CHANNELS_NORMALS * sizeof(float);
      return buffers;
    }

    void swapBackBuffer() override {
      std::swap(back_ptr, front_ptr);
      swap_sync.store(back_ptr, std::memory_order_relaxed);
    }

    FloatView backBuffer() override {
      return FloatView(back_ptr, render_buffers->getWidth() * render_buffers->getHeight() * RenderBufferPack::CHANNELS_COLOR);
    }
    FloatView frontBuffer() override {
      return FloatView(front_ptr, render_buffers->getWidth() * render_buffers->getHeight() * RenderBufferPack::CHANNELS_COLOR);
    }

    Framebuffer getFramebuffer() const override {
      Framebuffer output;
      output.color_buffer = FloatView(front_ptr, render_buffers->getWidth() * render_buffers->getHeight() * render_buffers->CHANNELS_COLOR);
      output.color_buffer_channels = render_buffers->CHANNELS_COLOR;
      output.color_buffer_width = render_buffers->getWidth();
      output.color_buffer_height = render_buffers->getHeight();

      output.depth_buffer = render_buffers->depthBuffer();
      output.depth_buffer_width = render_buffers->getWidth();
      output.depth_buffer_height = render_buffers->getHeight();

      output.normal_buffer = render_buffers->normalBuffer();
      output.normal_buffer_channels = render_buffers->CHANNELS_NORMALS;
      output.normal_buffer_width = render_buffers->getWidth();
      output.normal_buffer_height = render_buffers->getHeight();

      return output;
    }

    unsigned getWidth() const override { return render_buffers->getWidth(); }

    unsigned getHeight() const override { return render_buffers->getHeight(); }
  };

  RenderBufferPtr create_renderbuffer() { return std::make_unique<NvRenderBuffer>(); }
}  // namespace nova
