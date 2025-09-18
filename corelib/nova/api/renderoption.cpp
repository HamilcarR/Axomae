#include "private_includes.h"

namespace nova {
  class NvRenderOptions : public RenderOptions {
    unsigned aa_samples{};
    unsigned max_depth{};
    unsigned max_samples{};
    unsigned samples_increment{};
    unsigned tile_dimension_width{};
    unsigned tile_dimension_height{};
    bool flip_v{};
    bool uses_interops{false};
    bool uses_gpu{false};
    int integrator_flag{};

   public:
    NvRenderOptions() = default;

    NvRenderOptions(const RenderOptions &other) {
      aa_samples = other.getAliasingSamples();
      max_depth = other.getMaxDepth();
      max_samples = other.getMaxSamples();
      samples_increment = other.getSamplesIncrement();
      tile_dimension_width = other.getTileDimensionWidth();
      tile_dimension_height = other.getTileDimensionHeight();
      flip_v = other.isFlippedV();
      integrator_flag = other.getIntegratorFlag();
    }

    NvRenderOptions &operator=(const RenderOptions &other) {
      if (&other == this)
        return *this;
      aa_samples = other.getAliasingSamples();
      max_depth = other.getMaxDepth();
      max_samples = other.getMaxSamples();
      samples_increment = other.getSamplesIncrement();
      tile_dimension_width = other.getTileDimensionWidth();
      tile_dimension_height = other.getTileDimensionHeight();
      flip_v = other.isFlippedV();
      integrator_flag = other.getIntegratorFlag();
      return *this;
    }

    void setAliasingSamples(unsigned s) override { aa_samples = s; }

    void setMaxDepth(unsigned depth) override { max_depth = depth; }

    void setMaxSamples(unsigned samples) override { max_samples = samples; }

    void setSamplesIncrement(unsigned inc) override { samples_increment = inc; }

    void setTileDimension(unsigned width, unsigned height) override {
      tile_dimension_width = width;
      tile_dimension_height = height;
    }

    void flipV() override { flip_v = !flip_v; }

    bool isFlippedV() const override { return flip_v; }

    unsigned getAliasingSamples() const override { return aa_samples; }

    unsigned getSamplesIncrement() const override { return samples_increment; }

    unsigned getMaxDepth() const override { return max_depth; }

    unsigned getMaxSamples() const override { return max_samples; }

    unsigned getTileDimensionWidth() const override { return tile_dimension_width; }

    unsigned getTileDimensionHeight() const override { return tile_dimension_height; }

    ERROR_STATE useInterops(bool value) override {
      if (!core::build::is_gpu_build) {
        uses_interops = false;
        return NOT_GPU_BUILD;
      }
      uses_interops = value;
      return SUCCESS;
    }

    bool isUsingInterops() const override { return uses_interops; }

    ERROR_STATE useGpu(bool gpu) override {
      if (!core::build::is_gpu_build) {
        uses_gpu = false;
        return NOT_GPU_BUILD;
      }
      uses_gpu = gpu;
      return SUCCESS;
    }

    bool isUsingGpu() const override { return uses_gpu; }

    static ERROR_STATE check_valid_integrator_flags(int type) {
      int num_integrators = 0;

      /* Checks if user has chosen only one integrator ... cannot run Path + Metropolis at the same time for ex. */
      for (int i = 0; i <= 7; i++) {
        if ((type & (1 << i)) != 0)
          num_integrators++;
      }
      if (num_integrators >= 2)
        return MULTIPLE_INTEGRATORS_NOT_SUPPORTED;
      return SUCCESS;
    }

    ERROR_STATE setIntegratorFlag(int type) override {
      ERROR_STATE err{};
      if ((err = check_valid_integrator_flags(type)) == SUCCESS)
        integrator_flag = type;
      return err;
    }

    int getIntegratorFlag() const override { return integrator_flag; }
  };

  RenderOptionsPtr create_renderoptions() { return std::make_unique<NvRenderOptions>(); }
}  // namespace nova
