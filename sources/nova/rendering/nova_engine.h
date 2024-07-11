#ifndef NOVA_ENGINE_H
#define NOVA_ENGINE_H
#include "Axomae_macros.h"
#include "utils/nova_utils.h"
#include <atomic>
#include <map>
#include <string>

namespace nova {

  template<class T>
  struct RenderBuffers {
    T *accumulator_buffer;
    T *partial_buffer;
    size_t byte_size_buffers{};
    int channels{};
    std::vector<Tile> tiles;
  };
  using HdrBufferStruct = RenderBuffers<float>;

  namespace engine {
    class EngineResourcesHolder {
     private:
      int tiles_w{};
      int tiles_h{};
      int sample_increment{};
      int aliasing_samples{};
      int renderer_max_samples{};
      int max_depth{};
      std::atomic_long latency;
      bool *cancel_render;
      bool v_invert{false};
      std::string threadpool_tag;

     public:
      CLASS_CM(EngineResourcesHolder)

      void setTilesWidth(int width);
      void setTilesHeight(int height);
      void setSampleIncrement(int increment);
      void setAliasingSamples(int samples);
      void setMaxSamples(int samples);
      void setMaxDepth(int depth);
      void setCancelPtr(bool *cancel_ptr);
      void setVAxisInversed(bool invert);
      void setTag(const std::string &tag);
      [[nodiscard]] int getTilesWidth() const;
      [[nodiscard]] int getTilesHeight() const;
      [[nodiscard]] int getSampleIncrement() const;
      [[nodiscard]] int getAliasingSamples() const;
      [[nodiscard]] int getMaxSamples() const;
      [[nodiscard]] int getMaxDepth() const;
      [[nodiscard]] bool *getCancelPtr() const;
      [[nodiscard]] bool isAxisVInverted() const;
      [[nodiscard]] const std::string &getTag() const;
    };
  }  // namespace engine
}  // namespace nova

#endif
